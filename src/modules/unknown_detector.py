"""
文件名: unknown_detector.py
用途: 调用Mask2Anomaly模型并生成未知异常区域
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-24
"""

from time import perf_counter
from typing import Any
from typing import List
from typing import Optional

import cv2
import numpy as np

from src.interface.schemas import UnknownDetectionResult
from src.interface.schemas import UnknownRegion


INVALID_INPUT_ERROR = -1
MODEL_LOAD_ERROR = -2
INFERENCE_ERROR = -3
SUCCESS = 0


class UnknownDetector:
    """负责发现道路区域内的未知异常障碍物"""

    # 初始化未知障碍物检测配置和模型后端
    def __init__(
        self,
        config: dict,
        backend: Optional[Any] = None,
    ) -> None:
        self._backend_name = config.get(
            "backend",
            "mask2anomaly",
        )
        post_processing = config.get(
            "post_processing",
            {},
        )

        self._pixel_threshold = post_processing.get(
            "pixel_threshold",
            0.5,
        )
        self._min_area_ratio = post_processing.get(
            "min_area_ratio",
            0.0002,
        )
        self._max_area_ratio = post_processing.get(
            "max_area_ratio",
            0.15,
        )
        self._lower_roi_ratio = post_processing.get(
            "lower_roi_ratio",
            0.25,
        )
        self._roi_dilate_kernel_size = post_processing.get(
            "roi_dilate_kernel_size",
            31,
        )
        self._morphology_kernel_size = post_processing.get(
            "morphology_kernel_size",
            5,
        )
        self._region_score_quantile = post_processing.get(
            "region_score_quantile",
            0.95,
        )
        self._validate_configuration()
        self._backend = backend
        self._backend_error = ""
        if self._backend is None:
            self._initialize_backend(config)

    # 校验未知障碍物检测配置是否合法
    def _validate_configuration(self) -> None:
        if self._backend_name != "mask2anomaly":
            raise ValueError(
                f"Unsupported unknown detector backend: "
                f"{self._backend_name}"
            )
        if not 0.0 <= self._pixel_threshold <= 1.0:
            raise ValueError("pixel_threshold must be between 0 and 1.")
        if not 0.0 < self._min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1.")
        if not 0.0 < self._max_area_ratio <= 1.0:
            raise ValueError("max_area_ratio must be between 0 and 1.")
        if self._min_area_ratio > self._max_area_ratio:
            raise ValueError("min_area_ratio cannot exceed max_area_ratio.")
        if not 0.0 <= self._lower_roi_ratio < 1.0:
            raise ValueError("lower_roi_ratio must be in the range [0, 1).")
        if not 0.0 <= self._region_score_quantile <= 1.0:
            raise ValueError("region_score_quantile must be between 0 and 1.")
        self._validate_kernel_size(
            self._roi_dilate_kernel_size,
            "roi_dilate_kernel_size",
        )
        self._validate_kernel_size(
            self._morphology_kernel_size,
            "morphology_kernel_size",
        )

    # 校验形态学卷积核尺寸是否合法
    def _validate_kernel_size(
        self,
        kernel_size: int,
        config_name: str,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"{config_name} must be a positive odd integer."
            )

    # 初始化Mask2Anomaly模型后端
    def _initialize_backend(self, config: dict) -> None:
        try:
            from src.modules.mask2anomaly_backend import (Mask2AnomalyBackend,)
            self._backend = Mask2AnomalyBackend(config)
        except Exception as error:
            self._backend = None
            self._backend_error = (f"{type(error).__name__}: {error}")

    # 检查输入图像和道路掩码是否合法
    def _validate_input(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
    ) -> Optional[str]:
        if not isinstance(frame, np.ndarray):
            return "Input frame must be a NumPy array."
        if frame.size == 0:
            return "Input frame cannot be empty."
        if frame.ndim != 3 or frame.shape[2] != 3:
            return "Input frame must have shape H x W x 3."
        if frame.dtype != np.uint8:
            return "Input frame must use uint8 data type."
        if not isinstance(road_mask, np.ndarray):
            return "Road mask must be a NumPy array."
        if road_mask.ndim != 2:
            return "Road mask must have two dimensions."
        if road_mask.shape != frame.shape[:2]:
            return "Road mask shape must match input frame."
        return None

    # 创建失败状态下的空检测结果
    def _build_error_result(
        self,
        frame: np.ndarray,
        error_code: int,
        error_message: str,
        inference_time_ms: float = 0.0,
    ) -> UnknownDetectionResult:
        if isinstance(frame, np.ndarray) and frame.ndim >= 2:
            output_shape = frame.shape[:2]
        else:
            output_shape = (0, 0)
        score_map = np.zeros(
            output_shape,
            dtype=np.float32,
        )
        anomaly_mask = np.zeros(
            output_shape,
            dtype=np.uint8,
        )
        return UnknownDetectionResult(
            score_map=score_map,
            anomaly_mask=anomaly_mask,
            regions=[],
            inference_time_ms=inference_time_ms,
            error_code=error_code,
            error_message=error_message,
            model_version=self._get_model_version(),
        )

    # 获取模型版本或模型加载失败标识
    def _get_model_version(self) -> str:
        if self._backend is None:
            return "mask2anomaly-unavailable"
        return str(self._backend.model_version)

    # 构造道路异常检测感兴趣区域
    def _build_road_roi(
        self,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        height, width = road_mask.shape
        road_area = np.where(
            road_mask > 0,
            255,
            0,
        ).astype(np.uint8)
        roi_kernel = np.ones(
            (
                self._roi_dilate_kernel_size,
                self._roi_dilate_kernel_size,
            ),
            dtype=np.uint8,
        )
        road_roi = cv2.dilate(
            road_area,
            roi_kernel,
            iterations=1,
        )
        lower_roi = np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        roi_start = int(height * self._lower_roi_ratio)
        lower_roi[roi_start:, :] = 255
        return cv2.bitwise_and(
            road_roi,
            lower_roi,
        )

    # 清理阈值化后的异常候选区域
    def _clean_candidate_mask(
        self,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        kernel = np.ones(
            (
                self._morphology_kernel_size,
                self._morphology_kernel_size,
            ),
            dtype=np.uint8,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            kernel,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
        return candidate_mask

    # 将二值掩码编码为可写入JSON的游程格式
    def _encode_mask_rle(
        self,
        component_mask: np.ndarray,
    ) -> dict:
        flattened_mask = np.ravel(
            component_mask > 0,
            order="F",
        ).astype(np.uint8)
        change_indices = (
            np.flatnonzero(
                np.diff(flattened_mask),
            )
            + 1
        )
        run_starts = np.concatenate(
            (
                np.array([0]),
                change_indices,
            )
        )
        run_ends = np.concatenate(
            (
                change_indices,
                np.array([flattened_mask.size]),
            )
        )
        counts = (run_ends - run_starts).tolist()
        if flattened_mask.size > 0 and flattened_mask[0] == 1:
            counts.insert(0, 0)
        return {
            "size": [
                component_mask.shape[0],
                component_mask.shape[1],
            ],
            "counts": counts,
        }

    # 将异常掩码转换为未知区域列表
    def _mask_to_unknown_regions(
        self,
        anomaly_mask: np.ndarray,
        score_map: np.ndarray,
    ) -> List[UnknownRegion]:
        contours, _ = cv2.findContours(
            anomaly_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        image_area = anomaly_mask.shape[0] * anomaly_mask.shape[1]
        min_area = max(1, int(image_area * self._min_area_ratio),)
        max_area = max(min_area, int(image_area * self._max_area_ratio),)
        unknown_regions = []
        for region_index, contour in enumerate(contours):
            area = int(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            component_mask = np.zeros_like(anomaly_mask)
            cv2.drawContours(
                component_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )
            component_scores = score_map[component_mask > 0]
            if component_scores.size == 0:
                continue
            score = float(
                np.quantile(
                    component_scores,
                    self._region_score_quantile,
                )
            )
            x, y, box_width, box_height = cv2.boundingRect(contour)
            unknown_regions.append(
                UnknownRegion(
                    bbox=(
                        x,
                        y,
                        x + box_width,
                        y + box_height,
                    ),
                    score=score,
                    distance=None,
                    object_id=(
                        f"unknown-{region_index:03d}"
                    ),
                    area=area,
                    mask_rle=self._encode_mask_rle(
                        component_mask
                    ),
                )
            )
        return unknown_regions

    # 调用模型并生成未知障碍物检测结果
    def predict(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
    ) -> UnknownDetectionResult:
        start_time = perf_counter()
        input_error = self._validate_input(
            frame,
            road_mask,
        )
        if input_error is not None:
            return self._build_error_result(
                frame,
                INVALID_INPUT_ERROR,
                input_error,
            )
        if self._backend is None:
            return self._build_error_result(
                frame,
                MODEL_LOAD_ERROR,
                self._backend_error,
            )
        try:
            score_map, _ = self._backend.predict(frame)
            if score_map.shape != frame.shape[:2]:
                score_map = cv2.resize(
                    score_map,
                    (
                        frame.shape[1],
                        frame.shape[0],
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            score_map = np.ascontiguousarray(
                score_map,
                dtype=np.float32,
            )
            candidate_mask = np.where(
                score_map >= self._pixel_threshold,
                255,
                0,
            ).astype(np.uint8)
            road_roi = self._build_road_roi(road_mask)
            anomaly_mask = cv2.bitwise_and(
                candidate_mask,
                road_roi,
            )
            anomaly_mask = self._clean_candidate_mask(
                anomaly_mask
            )
            regions = self._mask_to_unknown_regions(
                anomaly_mask,
                score_map,
            )
            inference_time_ms = (
                perf_counter() - start_time
            ) * 1000.0
            return UnknownDetectionResult(
                score_map=score_map,
                anomaly_mask=anomaly_mask,
                regions=regions,
                inference_time_ms=inference_time_ms,
                error_code=SUCCESS,
                error_message="OK",
                model_version=self._get_model_version(),
            )
        except Exception as error:
            inference_time_ms = (
                perf_counter() - start_time
            ) * 1000.0
            return self._build_error_result(
                frame,
                INFERENCE_ERROR,
                f"{type(error).__name__}: {error}",
                inference_time_ms,
            )