"""
文件名: distance_estimator.py
用途: 对已知与未知障碍物做几何投影距离估计
作者: 周子懿
创建日期: 2026-07-28
最后修改日期: 2026-08-03
"""

from dataclasses import replace
from time import perf_counter
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from src.interface.module_interfaces import (
    DistanceEstimatorInterface,
)
from src.interface.schemas import DetectedObject
from src.interface.schemas import DistanceEstimationResult
from src.interface.schemas import UnknownRegion


INVALID_INPUT_ERROR = -1
INFERENCE_ERROR = -3
SUCCESS = 0


class DistanceEstimator(DistanceEstimatorInterface):
    """管理已知与未知障碍物的几何距离估计"""

    # 初始化距离估计器并解析配置参数
    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        self._enabled = bool(config.get("enabled", False))
        self._method = str(config.get("method", "not_implemented"))
        self._model_version = str(
            config.get(
                "model_version",
                "unimplemented",
            )
        )
        self._focal_length_px = float(config.get("focal_length_px", 1000.0))
        self._min_bbox_height_px = int(config.get("min_bbox_height_px", 8))
        self._min_distance_m = float(config.get("min_distance_m", 0.3))
        self._max_distance_m = float(config.get("max_distance_m", 80.0))
        self._camera_height_m = float(config.get("camera_height_m", 1.50))
        self._horizon_ratio = float(config.get("horizon_ratio", 0.50))
        default_heights = {
            "cone": 0.70,
            "barrier": 1.00,
            "vehicle": 1.50,
        }
        raw_heights = config.get(
            "class_heights_m",
            default_heights,
        )
        self._class_heights_m: Dict[str, float] = {
            str(name).lower().strip(): float(height)
            for name, height in raw_heights.items()
        }

    # 估计当前视频帧内所有已知障碍物的距离
    def estimate(
        self,
        frame: np.ndarray,
        frame_id: int,
        known_objects: List[DetectedObject],
    ) -> DistanceEstimationResult:
        start_time = perf_counter()

        if not self._enabled:
            return self._build_disabled_result(known_objects)

        input_error = self._validate_input(
            frame,
            known_objects,
        )
        if input_error is not None:
            return self._build_error_result(
                known_objects,
                INVALID_INPUT_ERROR,
                input_error,
            )

        try:
            estimated_objects = self._estimate_known_objects(
                frame=frame,
                frame_id=frame_id,
                known_objects=known_objects,
            )
            inference_time_ms = (perf_counter() - start_time) * 1000.0
            return DistanceEstimationResult(
                known_objects=estimated_objects,
                inference_time_ms=inference_time_ms,
                error_code=SUCCESS,
                error_message="OK",
                method=self._method,
                model_version=self._model_version,
                is_enabled=True,
            )
        except Exception as error:
            inference_time_ms = (perf_counter() - start_time) * 1000.0
            return self._build_error_result(
                known_objects,
                INFERENCE_ERROR,
                f"{type(error).__name__}: {error}",
                inference_time_ms,
            )

    # 校验输入帧和已知障碍物列表是否合法
    def _validate_input(
        self,
        frame: np.ndarray,
        known_objects: List[DetectedObject],
    ) -> Optional[str]:
        if not isinstance(frame, np.ndarray):
            return "Input frame must be a NumPy array."
        if frame.size == 0:
            return "Input frame cannot be empty."
        if frame.ndim != 3 or frame.shape[2] != 3:
            return "Input frame must have shape H x W x 3."
        if frame.dtype != np.uint8:
            return "Input frame must use uint8 data type."
        if not isinstance(known_objects, list):
            return "known_objects must be a list."
        return None

    # 构造模块关闭时的结果
    def _build_disabled_result(
        self,
        known_objects: List[DetectedObject],
    ) -> DistanceEstimationResult:
        return DistanceEstimationResult(
            known_objects=known_objects,
            inference_time_ms=0.0,
            error_code=SUCCESS,
            error_message=("Distance estimation is disabled."),
            method=self._method,
            model_version=self._model_version,
            is_enabled=False,
        )

    # 构造失败状态下的结果并保留原始检测
    def _build_error_result(
        self,
        known_objects: List[DetectedObject],
        error_code: int,
        error_message: str,
        inference_time_ms: float = 0.0,
    ) -> DistanceEstimationResult:
        return DistanceEstimationResult(
            known_objects=known_objects,
            inference_time_ms=inference_time_ms,
            error_code=error_code,
            error_message=error_message,
            method=self._method,
            model_version=self._model_version,
            is_enabled=True,
        )

    # 估计已知障碍物米制距离（几何投影）
    def _estimate_known_objects(
        self,
        frame: np.ndarray,
        frame_id: int,
        known_objects: List[DetectedObject],
    ) -> List[DetectedObject]:
        del frame_id
        frame_height, frame_width = frame.shape[:2]
        results: List[DetectedObject] = []

        for detected_object in known_objects:
            distance = self._estimate_one_object(
                detected_object=detected_object,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            results.append(
                replace(
                    detected_object,
                    distance=distance,
                )
            )
        return results

    # 使用区域底部接地点估计未知障碍物距离
    def estimate_unknown_regions(
        self,
        frame: np.ndarray,
        frame_id: int,
        unknown_regions: List[UnknownRegion],
    ) -> List[UnknownRegion]:
        del frame_id
        if not self._enabled:
            return unknown_regions
        if not isinstance(frame, np.ndarray):
            return unknown_regions
        if frame.size == 0 or frame.ndim != 3:
            return unknown_regions
        if not isinstance(unknown_regions, list):
            return []

        frame_height, frame_width = frame.shape[:2]
        estimated_regions: List[UnknownRegion] = []
        for region in unknown_regions:
            if not isinstance(region, UnknownRegion):
                continue
            distance = self._estimate_ground_distance(
                bbox=region.bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            estimated_regions.append(replace(region, distance=distance))
        return estimated_regions

    # 根据平坦路面投影计算边界框底部接地点距离
    def _estimate_ground_distance(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[float]:
        clipped_bbox = self._clip_bbox(
            bbox=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if clipped_bbox is None:
            return None
        if self._focal_length_px <= 0.0:
            return None
        if self._camera_height_m <= 0.0:
            return None
        if not 0.0 < self._horizon_ratio < 1.0:
            return None

        _x1, _y1, _x2, bottom_y = clipped_bbox
        horizon_y = frame_height * self._horizon_ratio
        vertical_offset = float(bottom_y) - horizon_y
        if vertical_offset <= 0.0:
            return None

        distance = (
            self._focal_length_px * self._camera_height_m / vertical_offset
        )
        if not (self._min_distance_m <= distance <= self._max_distance_m):
            return None
        return float(distance)

    # 对单个已知障碍物做几何距离估计
    def _estimate_one_object(
        self,
        detected_object: DetectedObject,
        frame_width: int,
        frame_height: int,
    ) -> Optional[float]:
        bbox = self._clip_bbox(
            detected_object.bbox,
            frame_width,
            frame_height,
        )
        if bbox is None:
            return None

        _x1, y1, _x2, y2 = bbox
        bbox_height = y2 - y1
        if bbox_height < self._min_bbox_height_px:
            return self._estimate_ground_distance(
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        class_key = detected_object.class_name.lower().strip()
        real_height = self._class_heights_m.get(class_key)
        if real_height is None or real_height <= 0.0:
            return self._estimate_ground_distance(
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        if self._focal_length_px <= 0.0:
            return None

        distance = self._focal_length_px * real_height / float(bbox_height)
        if not (self._min_distance_m <= distance <= self._max_distance_m):
            return None
        return float(distance)

    # 裁剪并校验边界框，无效时返回 None
    def _clip_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
