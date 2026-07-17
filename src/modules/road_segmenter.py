"""
文件名: road_segmenter.py
用途: 输出可行驶区域分割结果
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-17
"""

import logging
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.config.config import RoadSegmenterConfig
from src.interface.schemas import RoadSegmentResult, SystemStatus


class RoadSegmenter:
    """负责识别图像中的可行驶区域"""

    # 初始化RoadSegmenter
    def __init__(self, config: Optional[RoadSegmenterConfig] = None,
                 config_path: Optional[str] = None):
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
        self.processor = None
        self._model_initialized = False
        self._initialize_model()

    # 初始化计算设备
    def _init_device(self) -> torch.device:
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

    # 初始化HuggingFace SegFormer模型和图像处理器
    def _initialize_model(self) -> None:
        try:
            model_name = self.config.model.model_name

            self.logger.info(
                f"Loading SegFormer model from HuggingFace: {model_name}"
            )

            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=self.config.model.num_classes,
            )

            self.model.to(self.device)
            self.model.eval()

            self._model_initialized = True

            self.logger.info(
                f"SegFormer initialized successfully on {self.device}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize SegFormer: {str(e)}"
            )
            raise RuntimeError(
                f"Model initialization failed: {str(e)}"
            )

    # 将图像填充到尺寸能被divisor整除
    def _pad_to_divisible(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        divisor = self.config.data.divisor
        pad_height = (divisor - height % divisor) % divisor
        pad_width = (divisor - width % divisor) % divisor
        padded = np.pad(
            image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect"
        )
        return padded, pad_height, pad_width

    # 检测掩码边界
    def _detect_boundary(self, mask: np.ndarray) -> np.ndarray:
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

    # 平滑掩码
    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.post_processing.mask_smoothing:
            return mask

        kernel_size = self.config.post_processing.smoothing_kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    # 验证输入帧的有效性
    def _validate_input(self, frame: np.ndarray) -> Tuple[bool, int, str]:
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

    # 计算图像平均亮度
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    # 计算图像模糊度（拉普拉斯方差）
    def _calculate_blur(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    # 检查图像质量，判断降级状态
    def _check_image_quality(self, frame: np.ndarray) -> Tuple[str, dict]:
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

    # 对单帧图像进行可行驶区域分割
    def predict(self, frame: np.ndarray,
                timestamp: Optional[float] = None) -> RoadSegmentResult:
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            inputs = self.processor(
                images=frame_rgb,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=1)

                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                upsampled_probs = torch.nn.functional.interpolate(
                    probs,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

                predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()[0]

                road_confidence_map = upsampled_probs[
                    0, self.config.get_road_label()
                ].cpu().numpy()

            predicted_mask = predicted_mask.astype(np.uint8)

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

    # 获取模型信息
    def get_model_info(self) -> dict:
        return {
            "model_name": self.config.model.model_name,
            "num_classes": self.config.model.num_classes,
            "dataset": self.config.labels.dataset,
            "road_label": self.config.get_road_label(),
            "device": str(self.device),
            "model_initialized": self._model_initialized,
        }

    # 加载本地预训练权重（保留接口兼容性）
    def load_checkpoint(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.logger.info(f"Local checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}")
