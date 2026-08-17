"""
文件名: mask2anomaly_backend.py
用途: 加载Mask2Anomaly模型并输出像素级异常分数图
作者: 张楚涵
创建日期: 2026-07-24
最后修改日期: 2026-07-24
"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH_CHUNK_SIZE = 1024 * 1024


class Mask2AnomalyBackend:
    """负责加载Mask2Anomaly模型并执行像素级异常推理"""

    # 初始化模型配置、权重和推理器
    def __init__(self, config: dict) -> None:
        model_config = config["model"]
        post_processing_config = config.get(
            "post_processing",
            {},
        )
        inference_config = config.get(
            "inference",
            {},
        )
        self._device = model_config.get(
            "device",
            "cuda:0",
        )
        self._num_inlier_classes = model_config.get(
            "num_inlier_classes",
            19,
        )
        self._known_mask_threshold = post_processing_config.get(
            "known_mask_threshold",
            0.5,
        )
        self._enable_flip_tta = inference_config.get(
            "enable_flip_tta",
            False,
        )
        self._weights_sha256 = model_config.get(
            "weights_sha256",
            "",
        ).lower()
        self._config_path = self._resolve_project_file(
            model_config["config_path"],
            "Mask2Anomaly config",
        )
        self._weights_path = self._resolve_project_file(
            model_config["weights_path"],
            "Mask2Anomaly weights",
        )
        self._validate_configuration()
        self._verify_weights()
        self._predictor = self._build_predictor()
        self._model_version = (
            f"mask2anomaly-"
            f"{self._weights_path.stem}-"
            f"{self._weights_sha256[:12]}"
        )

    # 获取当前加载的模型版本
    @property
    def model_version(self) -> str:
        return self._model_version

    # 将项目相对路径解析为文件绝对路径
    def _resolve_project_file(
        self,
        relative_path: str,
        file_description: str,
    ) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(
                f"{file_description} path must be project-relative: "
                f"{relative_path}"
            )
        resolved_path = (PROJECT_ROOT / path).resolve()
        try:
            resolved_path.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                f"{file_description} path is outside project root: "
                f"{relative_path}"
            ) from error
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"{file_description} not found: {resolved_path}"
            )
        return resolved_path

    # 校验模型相关配置是否合法
    def _validate_configuration(self) -> None:
        if self._num_inlier_classes <= 0:
            raise ValueError("num_inlier_classes must be greater than zero.")
        if not 0.0 <= self._known_mask_threshold <= 1.0:
            raise ValueError("known_mask_threshold must be between 0 and 1.")
        if len(self._weights_sha256) != 64:
            raise ValueError("weights_sha256 must contain 64 characters.")
        if self._device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but is not available.")

    # 计算文件的SHA256校验值
    def _calculate_sha256(self, file_path: Path) -> str:
        digest = sha256()
        with file_path.open("rb") as file:
            while True:
                chunk = file.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    # 校验模型权重文件是否正确
    def _verify_weights(self) -> None:
        actual_sha256 = self._calculate_sha256(self._weights_path)
        if actual_sha256 != self._weights_sha256:
            raise RuntimeError(
                "Mask2Anomaly weights SHA256 mismatch. "
                f"Expected {self._weights_sha256}, "
                f"got {actual_sha256}."
            )

    # 根据项目配置创建Detectron2推理器
    def _build_predictor(self) -> DefaultPredictor:
        detectron_config = get_cfg()
        add_deeplab_config(detectron_config)
        add_maskformer2_config(detectron_config)
        detectron_config.merge_from_file(str(self._config_path))
        detectron_config.MODEL.WEIGHTS = str(self._weights_path)
        detectron_config.MODEL.DEVICE = self._device
        detectron_config.freeze()
        return DefaultPredictor(detectron_config)

    # 检查输入图像是否满足模型要求
    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("Input frame cannot be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")
        if frame.size == 0:
            raise ValueError("Input frame cannot be empty.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must have shape H x W x 3.")
        if frame.dtype != np.uint8:
            raise ValueError("Input frame must use uint8 data type.")

    # 等待CUDA任务完成以保证延迟统计准确
    def _synchronize_device(self) -> None:
        if self._device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self._device))

    # 将语义分割输出转换为异常分数图
    def _create_anomaly_score(
        self,
        sem_seg: torch.Tensor,
    ) -> np.ndarray:
        if sem_seg.ndim != 3:
            raise RuntimeError(
                "Mask2Anomaly sem_seg must have three dimensions."
            )
        if sem_seg.shape[0] < self._num_inlier_classes:
            raise RuntimeError(
                "Mask2Anomaly sem_seg has fewer channels than "
                "num_inlier_classes."
            )
        inlier_scores = sem_seg[: self._num_inlier_classes]
        anomaly_score = 1.0 - inlier_scores.amax(dim=0)
        extra_masks = sem_seg[self._num_inlier_classes :]
        if extra_masks.shape[0] > 0:
            known_mask = extra_masks.amax(dim=0)
            keep_mask = known_mask < self._known_mask_threshold
            anomaly_score = anomaly_score * keep_mask
        anomaly_score = (
            anomaly_score.clamp(0.0, 1.0).detach().float().cpu().numpy()
        )
        return np.ascontiguousarray(anomaly_score)

    # 对单张图像执行一次Mask2Anomaly推理
    def _predict_single(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        contiguous_frame = np.ascontiguousarray(frame)
        outputs = self._predictor(contiguous_frame)
        if "sem_seg" not in outputs:
            raise RuntimeError("Mask2Anomaly output does not contain sem_seg.")
        return self._create_anomaly_score(outputs["sem_seg"])

    # 执行异常推理并返回异常分数图和推理耗时
    @torch.inference_mode()
    def predict(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        self._validate_frame(frame)
        self._synchronize_device()
        start_time = perf_counter()
        anomaly_score = self._predict_single(frame)
        if self._enable_flip_tta:
            flipped_frame = np.ascontiguousarray(np.fliplr(frame))
            flipped_score = self._predict_single(flipped_frame)
            flipped_score = np.fliplr(flipped_score)
            anomaly_score = (anomaly_score + flipped_score) / 2.0
            anomaly_score = np.ascontiguousarray(
                anomaly_score,
                dtype=np.float32,
            )
        self._synchronize_device()
        inference_time_ms = (perf_counter() - start_time) * 1000.0
        return anomaly_score, inference_time_ms
