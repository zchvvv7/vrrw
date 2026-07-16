"""
文件名: road_segmenter.py
用途: 输出可行驶区域分割结果
作者:温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import cv2
import numpy as np
import torch
import time
import logging
import os
from typing import Optional, Tuple

import segmentation_models_pytorch as smp

from src.config.config import RoadSegmenterConfig
from src.interface.schemas import RoadSegmentResult, SystemStatus


class RoadSegmenter:
    """负责识别图像中的可行驶区域"""

    def __init__(self, config: Optional[RoadSegmenterConfig] = None,
                 config_path: Optional[str] = None):
        """
        初始化RoadSegmenter

        Args:
            config: 配置对象，若为None则从config_path加载或使用默认配置
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = RoadSegmenterConfig.from_yaml(config_path)
        else:
            self.config = RoadSegmenterConfig()

        self.device = self._init_device()
        self.model = None
        self.preprocess_fn = None
        self._model_initialized = False
        self._initialize_model()

    def _init_device(self) -> torch.device:
        """
        初始化计算设备

        Returns:
            torch.device实例
        """
        device_config = self.config.performance.device
        if device_config == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _initialize_model(self) -> None:
        """初始化SegFormer模型"""
        try:
            self.model = smp.Segformer(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=3,
                classes=self.config.model.num_classes,
            )
            self.model.to(self.device)
            self.model.eval()

            if self.config.model.checkpoint_path is not None:
                if os.path.exists(self.config.model.checkpoint_path):
                    self.load_checkpoint(self.config.model.checkpoint_path)
                else:
                    self.logger.warning(
                        f"Checkpoint not found at {self.config.model.checkpoint_path}, "
                        f"using randomly initialized decoder. "
                        f"Run 'python scripts/download_checkpoint.py' to download "
                        f"the pretrained weights."
                    )

            self.preprocess_fn = smp.encoders.get_preprocessing_fn(
                self.config.model.encoder_name,
                self.config.model.encoder_weights,
            )
            self._model_initialized = True
            self.logger.info(f"Model initialized successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _remap_checkpoint_keys(self, state_dict: dict) -> dict:
        """
        重映射MMSegmentation格式的权重key到SMP格式

        Args:
            state_dict: 原始权重字典

        Returns:
            重映射后的权重字典
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "encoder.", 1)
            elif key.startswith("decode_head."):
                new_key = key.replace("decode_head.", "decoder.", 1)
            elif key.startswith("seg_head."):
                new_key = key.replace("seg_head.", "decoder.", 1)
            new_state_dict[new_key] = value
        return new_state_dict

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载预训练模型权重

        Args:
            checkpoint_path: 权重文件路径

        Raises:
            RuntimeError: 加载失败
        """
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            state_dict = self._remap_checkpoint_keys(state_dict)

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}")

    def _pad_to_divisible(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        将图像填充到尺寸能被divisor整除

        Args:
            image: 输入图像

        Returns:
            (填充后的图像, 填充高度, 填充宽度)
        """
        height, width = image.shape[:2]
        divisor = self.config.data.divisor
        pad_height = (divisor - height % divisor) % divisor
        pad_width = (divisor - width % divisor) % divisor
        padded = np.pad(
            image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect"
        )
        return padded, pad_height, pad_width

    def _detect_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        检测掩码边界

        Args:
            mask: 二值掩码

        Returns:
            边界图像，若未启用边界检测则返回None
        """
        if not self.config.post_processing.boundary_detection:
            return None

        method = self.config.post_processing.boundary_method
        if method == "canny":
            low = self.config.post_processing.canny_low_threshold
            high = self.config.post_processing.canny_high_threshold
            return cv2.Canny(mask, low, high)
        elif method == "sobel":
            sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))
        else:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boundary = np.zeros_like(mask)
            cv2.drawContours(boundary, contours, -1, 255, 2)
            return boundary

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        平滑掩码

        Args:
            mask: 二值掩码

        Returns:
            平滑后的掩码
        """
        if not self.config.post_processing.mask_smoothing:
            return mask

        kernel_size = self.config.post_processing.smoothing_kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _validate_input(self, frame: np.ndarray) -> Tuple[bool, int, str]:
        """
        验证输入帧的有效性

        Args:
            frame: 输入图像

        Returns:
            (是否有效, 错误码, 错误信息)
        """
        if frame is None:
            return (
                False,
                self.config.system.error_codes["invalid_input"],
                "Input frame is None"
            )
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return (
                False,
                self.config.system.error_codes["invalid_input"],
                f"Invalid input shape: {frame.shape}"
            )
        if frame.size == 0:
            return (
                False,
                self.config.system.error_codes["invalid_input"],
                "Input frame is empty"
            )
        return (
            True,
            self.config.system.error_codes["success"],
            "OK"
        )

    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """
        计算图像平均亮度

        Args:
            frame: 输入图像

        Returns:
            平均亮度值（0-255）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _calculate_blur(self, frame: np.ndarray) -> float:
        """
        计算图像模糊度（拉普拉斯方差）

        Args:
            frame: 输入图像

        Returns:
            模糊度值（方差）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    def _check_image_quality(self, frame: np.ndarray) -> Tuple[str, dict]:
        """
        检查图像质量，判断降级状态

        Args:
            frame: 输入图像

        Returns:
            (系统状态, 质量指标字典)
        """
        if not self.config.quality.enable_quality_check:
            return SystemStatus.NORMAL, {}

        brightness = self._calculate_brightness(frame)
        blur = self._calculate_blur(frame)

        quality_metrics = {
            "brightness": brightness,
            "blur": blur,
        }

        degraded_reasons = []
        unavailable_reasons = []

        if brightness < self.config.quality.dark_threshold:
            unavailable_reasons.append(
                f"too_dark (brightness={brightness:.1f})"
            )
        elif brightness > self.config.quality.bright_threshold:
            degraded_reasons.append(
                f"too_bright (brightness={brightness:.1f})"
            )
        elif brightness < self.config.quality.brightness_low_threshold:
            degraded_reasons.append(
                f"low_brightness (brightness={brightness:.1f})"
            )
        elif brightness > self.config.quality.brightness_high_threshold:
            degraded_reasons.append(
                f"high_brightness (brightness={brightness:.1f})"
            )

        if blur < self.config.quality.blur_threshold:
            degraded_reasons.append(f"blurry (blur={blur:.1f})")

        quality_metrics["degraded_reasons"] = degraded_reasons
        quality_metrics["unavailable_reasons"] = unavailable_reasons

        if unavailable_reasons:
            return SystemStatus.UNAVAILABLE, quality_metrics
        elif degraded_reasons:
            return SystemStatus.DEGRADED, quality_metrics
        else:
            return SystemStatus.NORMAL, quality_metrics

    def predict(self, frame: np.ndarray,
                timestamp: Optional[float] = None) -> RoadSegmentResult:
        """
        对单帧图像进行可行驶区域分割

        Args:
            frame: 输入图像（BGR格式）
            timestamp: 时间戳

        Returns:
            RoadSegmentResult分割结果
        """
        start_time = time.time()

        valid, error_code, error_msg = self._validate_input(frame)
        if not valid:
            height, width = (
                frame.shape[:2]
                if frame is not None and len(frame.shape) >= 2
                else (0, 0)
            )
            return RoadSegmentResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boundary=None,
                confidence_map=None,
                global_confidence=0.0,
                road_pixel_ratio=0.0,
                error_code=error_code,
                error_message=error_msg,
                inference_time_ms=0.0,
                timestamp=timestamp,
                system_status=SystemStatus.UNAVAILABLE,
                quality_metrics={},
            )

        system_status, quality_metrics = self._check_image_quality(frame)

        if system_status == SystemStatus.UNAVAILABLE:
            height, width = frame.shape[:2]
            reason = ", ".join(
                quality_metrics.get("unavailable_reasons", [])
            )
            self.logger.warning(
                f"Image quality check failed - unavailable: {reason}"
            )
            return RoadSegmentResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boundary=None,
                confidence_map=None,
                global_confidence=0.0,
                road_pixel_ratio=0.0,
                error_code=self.config.system.error_codes["success"],
                error_message=f"Image quality unavailable: {reason}",
                inference_time_ms=(time.time() - start_time) * 1000,
                timestamp=timestamp,
                system_status=SystemStatus.UNAVAILABLE,
                quality_metrics=quality_metrics,
            )

        try:
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

            input_height, input_width = self.config.data.input_size
            resized_frame = cv2.resize(
                frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR
            )

            padded, pad_height, pad_width = self._pad_to_divisible(resized_frame)

            frame_preprocessed = padded / 255.0
            if self.config.data.normalize:
                frame_preprocessed = (
                    frame_preprocessed - np.array(self.config.data.mean)
                ) / np.array(self.config.data.std)

            frame_preprocessed = np.transpose(
                frame_preprocessed, (2, 0, 1)
            ).astype(np.float32)
            frame_tensor = torch.from_numpy(frame_preprocessed).unsqueeze(0).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(frame_tensor)
                logits = outputs
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = (
                        outputs.logits
                        if hasattr(outputs, "logits")
                        else outputs[0]
                    )

                probs = torch.softmax(logits, dim=1)

                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(input_height + pad_height, input_width + pad_width),
                    mode="bilinear",
                    align_corners=False,
                )
                upsampled_probs = torch.nn.functional.interpolate(
                    probs,
                    size=(input_height + pad_height, input_width + pad_width),
                    mode="bilinear",
                    align_corners=False,
                )

                if pad_height > 0:
                    upsampled_logits = upsampled_logits[:, :, :-pad_height, :]
                    upsampled_probs = upsampled_probs[:, :, :-pad_height, :]
                if pad_width > 0:
                    upsampled_logits = upsampled_logits[:, :, :, :-pad_width]
                    upsampled_probs = upsampled_probs[:, :, :, :-pad_width]

                predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
                road_confidence_map = upsampled_probs[
                    0, self.config.get_road_label()
                ].cpu().numpy()

            predicted_mask = cv2.resize(
                predicted_mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            road_confidence_map = cv2.resize(
                road_confidence_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            road_mask = np.where(
                predicted_mask == self.config.get_road_label(), 255, 0
            ).astype(np.uint8)
            road_mask = self._smooth_mask(road_mask)

            boundary = self._detect_boundary(road_mask)

            road_pixels = np.sum(road_mask == 255)
            total_pixels = height * width
            road_pixel_ratio = (
                road_pixels / total_pixels if total_pixels > 0 else 0.0
            )

            conf_threshold = self.config.post_processing.confidence_threshold
            confident_road_pixels = np.sum(
                road_confidence_map[road_mask == 255] >= conf_threshold
            )
            global_confidence = (
                confident_road_pixels / road_pixels
                if road_pixels > 0
                else 0.0
            )

            if (self.config.quality.enable_quality_check
                    and system_status == SystemStatus.NORMAL):
                if global_confidence < self.config.quality.degraded_confidence_threshold:
                    system_status = SystemStatus.DEGRADED
                    quality_metrics["degraded_reasons"] = (
                        quality_metrics.get("degraded_reasons", [])
                    )
                    quality_metrics["degraded_reasons"].append(
                        f"low_confidence ({global_confidence:.3f})"
                    )
                elif global_confidence < (
                    self.config.quality.unavailable_confidence_threshold
                ):
                    system_status = SystemStatus.UNAVAILABLE
                    quality_metrics["unavailable_reasons"] = (
                        quality_metrics.get("unavailable_reasons", [])
                    )
                    quality_metrics["unavailable_reasons"].append(
                        f"very_low_confidence ({global_confidence:.3f})"
                    )

            if system_status == SystemStatus.DEGRADED:
                reason = ", ".join(
                    quality_metrics.get("degraded_reasons", [])
                )
                self.logger.warning(f"Image quality degraded: {reason}")

            inference_time_ms = (time.time() - start_time) * 1000

            return RoadSegmentResult(
                mask=road_mask,
                boundary=boundary,
                confidence_map=road_confidence_map,
                global_confidence=global_confidence,
                road_pixel_ratio=road_pixel_ratio,
                error_code=self.config.system.error_codes["success"],
                error_message=(
                    "OK"
                    if system_status == SystemStatus.NORMAL
                    else f"Degraded: {', '.join(quality_metrics.get('degraded_reasons', []))}"
                ),
                inference_time_ms=inference_time_ms,
                timestamp=timestamp,
                system_status=system_status,
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            self.logger.error(f"Inference error: {str(e)}")
            return RoadSegmentResult(
                mask=np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8),
                boundary=None,
                confidence_map=None,
                global_confidence=0.0,
                road_pixel_ratio=0.0,
                error_code=self.config.system.error_codes["inference_error"],
                error_message=str(e),
                inference_time_ms=(time.time() - start_time) * 1000,
                timestamp=timestamp,
                system_status=SystemStatus.UNAVAILABLE,
                quality_metrics={},
            )

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "encoder_name": self.config.model.encoder_name,
            "num_classes": self.config.model.num_classes,
            "dataset": self.config.labels.dataset,
            "road_label": self.config.get_road_label(),
            "device": str(self.device),
            "model_initialized": self._model_initialized,
        }