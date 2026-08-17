"""
文件名: known_detector.py
用途: 检测已知类型道路障碍物
作者: 周子懿
创建日期: 2026-07-16
最后修改日期: 2026-07-27
"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import List
from typing import Optional
from typing import Set

import numpy as np
import torch

from src.interface.schemas import DetectedObject
from src.interface.schemas import KnownDetectionResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH_CHUNK_SIZE = 1024 * 1024
INVALID_INPUT_ERROR = -1
MODEL_LOAD_ERROR = -2
INFERENCE_ERROR = -3
SUCCESS = 0


class KnownDetector:
    """负责检测锥桶、护栏、坑洞、车辆等已知障碍物"""

    # 初始化已知障碍物检测器并加载YOLO模型
    def __init__(
        self,
        config: dict,
        model: Optional[Any] = None,
    ) -> None:
        self._model_path_value = str(
            config.get(
                "model_name",
                "checkpoints/yolo_best.pt",
            )
        )
        self._conf_threshold = float(config.get("conf_threshold", 0.25))
        self._iou_threshold = float(config.get("iou_threshold", 0.45))
        self._device = config.get("device", "cpu")
        class_list = config.get("classes", [])
        self._target_classes: Set[str] = {
            name.lower().strip() for name in class_list
        }
        self._weights_sha256 = str(config.get("weights_sha256", "")).lower()
        self._model = model
        self._model_error = ""
        self._model_path: Optional[Path] = None
        self._model_version = "yolo-unavailable"

        self._initialize_model()

    # 获取当前已知障碍物模型版本
    @property
    def model_version(self) -> str:
        return self._model_version

    # 初始化模型配置、权重和YOLO推理器
    def _initialize_model(self) -> None:
        try:
            self._validate_configuration()
            self._model_path = self._resolve_model_path(self._model_path_value)
            self._verify_weights()

            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(str(self._model_path))

            self._validate_model_classes()
            version_suffix = self._weights_sha256[:12]
            if not version_suffix:
                version_suffix = "unchecked"
            self._model_version = (
                f"yolo-{self._model_path.stem}-{version_suffix}"
            )
        except Exception as error:
            self._model = None
            self._model_error = f"{type(error).__name__}: {error}"

    # 校验检测阈值、设备和类别配置
    def _validate_configuration(self) -> None:
        if not self._model_path_value:
            raise ValueError("model_name cannot be empty.")

        if not 0.0 <= self._conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1.")

        if not 0.0 <= self._iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        if isinstance(self._device, str) and self._device.lower() == "auto":
            self._device = 0 if torch.cuda.is_available() else "cpu"

        if not isinstance(self._device, (str, int)):
            raise TypeError("device must be a string or integer.")

        if "" in self._target_classes:
            raise ValueError("classes cannot contain an empty name.")

        if self._weights_sha256 and len(self._weights_sha256) != 64:
            raise ValueError("weights_sha256 must contain 64 characters.")

    # 将项目相对模型路径解析为绝对路径
    def _resolve_model_path(
        self,
        model_path_value: str,
    ) -> Path:
        model_path = Path(model_path_value)
        if model_path.is_absolute():
            raise ValueError(
                "Known detector model path must be project-relative."
            )

        resolved_path = (PROJECT_ROOT / model_path).resolve()
        try:
            resolved_path.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                "Known detector model path is outside project root."
            ) from error

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Known detector model not found: {resolved_path}"
            )

        return resolved_path

    # 计算模型权重的SHA256校验值
    def _calculate_sha256(
        self,
        file_path: Path,
    ) -> str:
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
        if not self._weights_sha256:
            return

        if self._model_path is None:
            raise RuntimeError("Known detector model path is unavailable.")

        actual_sha256 = self._calculate_sha256(self._model_path)
        if actual_sha256 != self._weights_sha256:
            raise RuntimeError(
                "Known detector weights SHA256 mismatch. "
                f"Expected {self._weights_sha256}, "
                f"got {actual_sha256}."
            )

    # 校验配置类别是否存在于模型类别中
    def _validate_model_classes(self) -> None:
        if self._model is None:
            raise RuntimeError("Known detector model is unavailable.")

        model_names = getattr(
            self._model,
            "names",
            None,
        )
        if model_names is None:
            raise RuntimeError("Known detector model has no class names.")

        if isinstance(model_names, dict):
            available_classes = {
                str(name).lower() for name in model_names.values()
            }
        else:
            available_classes = {str(name).lower() for name in model_names}

        missing_classes = self._target_classes - available_classes
        if missing_classes:
            missing_text = ", ".join(sorted(missing_classes))
            raise ValueError(
                f"Configured classes are absent from model: {missing_text}"
            )

    # 检查输入图像是否满足检测要求
    def _validate_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[str]:
        if not isinstance(frame, np.ndarray):
            return "Input frame must be a NumPy array."

        if frame.size == 0:
            return "Input frame cannot be empty."

        if frame.ndim != 3 or frame.shape[2] != 3:
            return "Input frame must have shape H x W x 3."

        if frame.dtype != np.uint8:
            return "Input frame must use uint8 data type."

        return None

    # 创建失败状态下的空检测结果
    def _build_error_result(
        self,
        error_code: int,
        error_message: str,
        inference_time_ms: float = 0.0,
    ) -> KnownDetectionResult:
        return KnownDetectionResult(
            objects=[],
            inference_time_ms=inference_time_ms,
            error_code=error_code,
            error_message=error_message,
            model_version=self._model_version,
        )

    # 将YOLO检测框转换为项目统一数据结构
    def _convert_detections(
        self,
        results: Any,
    ) -> List[DetectedObject]:
        detected_objects: List[DetectedObject] = []
        if not results:
            return detected_objects

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detected_objects

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id]).lower()
            if self._target_classes and class_name not in self._target_classes:
                continue

            xyxy = box.xyxy[0].tolist()
            detected_objects.append(
                DetectedObject(
                    class_name=class_name,
                    bbox=(
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                    ),
                    confidence=float(box.conf[0].item()),
                    distance=None,
                )
            )

        return detected_objects

    # 对单帧图像执行已知障碍物检测
    def predict(
        self,
        frame: np.ndarray,
    ) -> KnownDetectionResult:
        start_time = perf_counter()
        input_error = self._validate_frame(frame)

        if input_error is not None:
            return self._build_error_result(
                INVALID_INPUT_ERROR,
                input_error,
            )

        if self._model is None:
            return self._build_error_result(
                MODEL_LOAD_ERROR,
                self._model_error,
            )

        try:
            results = self._model.predict(
                source=frame,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                device=self._device,
                verbose=False,
            )
            detected_objects = self._convert_detections(results)
            inference_time_ms = (perf_counter() - start_time) * 1000.0
            return KnownDetectionResult(
                objects=detected_objects,
                inference_time_ms=inference_time_ms,
                error_code=SUCCESS,
                error_message="OK",
                model_version=self._model_version,
            )
        except Exception as error:
            inference_time_ms = (perf_counter() - start_time) * 1000.0
            return self._build_error_result(
                INFERENCE_ERROR,
                f"{type(error).__name__}: {error}",
                inference_time_ms,
            )
