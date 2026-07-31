"""
文件名: corridor_predictor.py
用途: 预测视频图像空间中的自车行驶走廊，检测碰撞风险，给出方向建议
作者: 温涵清
创建日期: 2026-07-28
最后修改日期: 2026-07-30
"""

from collections import deque
from time import perf_counter
from typing import Deque
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from src.interface.module_interfaces import CorridorPredictorInterface
from src.interface.schemas import CorridorPredictionResult


INVALID_INPUT_ERROR = -1
MODEL_LOAD_ERROR = -2
INFERENCE_ERROR = -3
SUCCESS = 0

# 历史结果最大缓存帧数
MAX_HISTORY_SIZE = 10


class CorridorPredictor(CorridorPredictorInterface):
    """预测自车行驶走廊，检测障碍物碰撞风险，建议行驶方向"""

    # 初始化走廊预测器并解析配置参数
    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        self._enabled = bool(config.get("enabled", False))
        self._method = str(
            config.get("method", "road_mask_geometry")
        )
        self._model_version = str(
            config.get("model_version", "v1.0")
        )
        self._min_road_width = int(
            config.get("min_road_width", 50)
        )
        self._max_centerline_points = int(
            config.get("max_centerline_points", 200)
        )
        self._temporal_window = int(
            config.get("temporal_window", 5)
        )
        self._confidence_decay = float(
            config.get("confidence_decay", 0.9)
        )
        self._boundary_sample_step = int(
            config.get("boundary_sample_step", 2)
        )
        self._min_valid_rows = int(
            config.get("min_valid_rows", 10)
        )
        self._smoothing_kernel = int(
            config.get("smoothing_kernel", 5)
        )
        self._obstacle_edge_threshold = int(
            config.get("obstacle_edge_threshold", 50)
        )
        self._obstacle_min_area_ratio = float(
            config.get("obstacle_min_area_ratio", 0.001)
        )
        self._obstacle_risk_weight = float(
            config.get("obstacle_risk_weight", 0.5)
        )
        self._direction_risk_weight = float(
            config.get("direction_risk_weight", 0.3)
        )
        self._width_risk_weight = float(
            config.get("width_risk_weight", 0.2)
        )
        self._history_buffer: Deque[dict] = deque(
            maxlen=MAX_HISTORY_SIZE
        )

    # 清空所有跨帧历史状态
    def reset(self) -> None:
        self._history_buffer.clear()

    # 预测当前帧的自车行驶走廊
    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
    ) -> CorridorPredictionResult:
        start_time = perf_counter()

        if not self._enabled:
            return self._build_disabled_result()

        input_error = self._validate_input(frame, road_mask)
        if input_error is not None:
            return self._build_error_result(
                frame,
                INVALID_INPUT_ERROR,
                input_error,
            )

        try:
            (
                corridor_mask,
                polygon,
                centerline,
                confidence,
            ) = self._predict_corridor(
                frame=frame,
                frame_id=frame_id,
                road_mask=road_mask,
            )

            inference_time_ms = (
                perf_counter() - start_time
            ) * 1000.0

            return CorridorPredictionResult(
                corridor_mask=corridor_mask,
                polygon=polygon,
                centerline=centerline,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
                error_code=SUCCESS,
                error_message="OK",
                method=self._method,
                model_version=self._model_version,
                is_enabled=True,
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

    # 校验输入帧和道路掩码是否合法
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

    # 构造模块关闭时的空结果
    def _build_disabled_result(self) -> CorridorPredictionResult:
        return CorridorPredictionResult(
            corridor_mask=None,
            polygon=[],
            centerline=[],
            confidence=0.0,
            inference_time_ms=0.0,
            error_code=SUCCESS,
            error_message="Corridor prediction is disabled.",
            method=self._method,
            model_version=self._model_version,
            is_enabled=False,
        )

    # 构造失败状态下的空结果
    def _build_error_result(
        self,
        frame: np.ndarray,
        error_code: int,
        error_message: str,
        inference_time_ms: float = 0.0,
    ) -> CorridorPredictionResult:
        if isinstance(frame, np.ndarray) and frame.ndim >= 2:
            height, width = frame.shape[:2]
        else:
            height, width = 0, 0
        return CorridorPredictionResult(
            corridor_mask=None,
            polygon=[],
            centerline=[],
            confidence=0.0,
            inference_time_ms=inference_time_ms,
            error_code=error_code,
            error_message=error_message,
            method=self._method,
            model_version=self._model_version,
            is_enabled=True,
        )

    # 核心算法：从道路掩码中提取走廊、检测风险、计算置信度
    def _predict_corridor(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]],
               List[Tuple[int, int]], float]:
        height, width = road_mask.shape[:2]

        left_boundary, right_boundary, valid_rows = (
            self._extract_road_boundaries(road_mask)
        )

        if len(valid_rows) < self._min_valid_rows:
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            self._update_history(
                empty_mask, [], [], 0.0
            )
            return empty_mask, [], [], 0.0

        centerline = self._fit_centerline(
            left_boundary, right_boundary, valid_rows
        )

        centerline = self._smooth_centerline(centerline)

        corridor_polygon = self._build_corridor_polygon(
            centerline, left_boundary, right_boundary, valid_rows
        )

        corridor_mask = self._build_corridor_mask(
            corridor_polygon, height, width
        )

        corridor_mask = self._refine_corridor_with_obstacles(
            corridor_mask, frame, road_mask
        )

        direction, direction_risk = self._analyze_direction(
            centerline, height, width
        )

        obstacle_risk = self._estimate_obstacle_risk(
            corridor_mask, frame, height, width
        )

        width_risk = self._estimate_width_risk(
            corridor_mask, road_mask
        )

        raw_confidence = self._compute_confidence(
            obstacle_risk, direction_risk, width_risk
        )

        smoothed_confidence = self._temporal_smooth_confidence(
            raw_confidence
        )

        centerline = self._densify_centerline(centerline)

        self._update_history(
            corridor_mask,
            corridor_polygon,
            centerline,
            smoothed_confidence,
        )

        return (
            corridor_mask,
            corridor_polygon,
            centerline,
            smoothed_confidence,
        )

    # 从道路掩码中逐行提取左右边界
    def _extract_road_boundaries(
        self,
        road_mask: np.ndarray,
    ) -> Tuple[dict, dict, List[int]]:
        height, width = road_mask.shape[:2]
        left_boundary: dict = {}
        right_boundary: dict = {}
        valid_rows: List[int] = []

        for y in range(0, height, self._boundary_sample_step):
            row_pixels = np.where(road_mask[y, :] > 0)[0]
            if len(row_pixels) >= 2:
                left_boundary[y] = int(row_pixels[0])
                right_boundary[y] = int(row_pixels[-1])
                valid_rows.append(y)

        if len(valid_rows) >= 2:
            left_boundary = self._interpolate_boundaries(
                left_boundary, valid_rows, height, width
            )
            right_boundary = self._interpolate_boundaries(
                right_boundary, valid_rows, height, width
            )

        return left_boundary, right_boundary, valid_rows

    # 对边界缺失的行进行线性插值填充
    def _interpolate_boundaries(
        self,
        boundary: dict,
        valid_rows: List[int],
        height: int,
        width: int,
    ) -> dict:
        if len(valid_rows) < 2:
            return boundary

        new_boundary = dict(boundary)
        all_rows = sorted(
            set(range(0, height, self._boundary_sample_step))
        )
        sorted_valid = sorted(valid_rows)

        for y in all_rows:
            if y in new_boundary:
                continue
            prev_valid = None
            next_valid = None
            for vy in sorted_valid:
                if vy < y:
                    prev_valid = vy
                if vy > y and next_valid is None:
                    next_valid = vy
                    break
            if prev_valid is not None and next_valid is not None:
                ratio = (y - prev_valid) / (
                    next_valid - prev_valid
                )
                val = int(
                    new_boundary.get(prev_valid, 0)
                    + ratio
                    * (
                        new_boundary.get(next_valid, 0)
                        - new_boundary.get(prev_valid, 0)
                    )
                )
                new_boundary[y] = max(0, min(width - 1, val))

        return new_boundary

    # 基于左右边界计算中心线
    def _fit_centerline(
        self,
        left_boundary: dict,
        right_boundary: dict,
        valid_rows: List[int],
    ) -> List[Tuple[int, int]]:
        centerline: List[Tuple[int, int]] = []
        for y in sorted(valid_rows):
            lx = left_boundary.get(y)
            rx = right_boundary.get(y)
            if lx is not None and rx is not None:
                cx = int((lx + rx) / 2)
                centerline.append((cx, y))
        return centerline

    # 对中心线进行滑动平均平滑
    def _smooth_centerline(
        self,
        centerline: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if len(centerline) < 3:
            return centerline

        window = min(3, len(centerline))
        smoothed: List[Tuple[int, int]] = []
        half_w = window // 2

        for i in range(len(centerline)):
            start = max(0, i - half_w)
            end = min(len(centerline), i + half_w + 1)
            window_points = centerline[start:end]
            avg_x = int(
                sum(p[0] for p in window_points)
                / len(window_points)
            )
            avg_y = int(
                sum(p[1] for p in window_points)
                / len(window_points)
            )
            smoothed.append((avg_x, avg_y))

        return smoothed

    # 构建走廊多边形（底部→顶部→返回底部）
    def _build_corridor_polygon(
        self,
        centerline: List[Tuple[int, int]],
        left_boundary: dict,
        right_boundary: dict,
        valid_rows: List[int],
    ) -> List[Tuple[int, int]]:
        if not centerline:
            return []

        sorted_rows = sorted(valid_rows)
        left_points: List[Tuple[int, int]] = []
        right_points: List[Tuple[int, int]] = []

        for y in sorted_rows:
            lx = left_boundary.get(y)
            rx = right_boundary.get(y)
            if lx is not None:
                left_points.append((lx, y))
            if rx is not None:
                right_points.append((rx, y))

        polygon: List[Tuple[int, int]] = []
        for p in reversed(left_points):
            polygon.append(p)
        for p in right_points:
            polygon.append(p)

        return polygon

    # 用多边形填充生成走廊掩码
    def _build_corridor_mask(
        self,
        polygon: List[Tuple[int, int]],
        height: int,
        width: int,
    ) -> np.ndarray:
        if len(polygon) < 3:
            return np.zeros((height, width), dtype=np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)
        contour = np.array(polygon, dtype=np.int32).reshape(
            (-1, 1, 2)
        )
        cv2.fillPoly(mask, [contour], 255)
        return mask

    # 在走廊底部区域检测明显障碍物并扣除碰撞区域
    def _refine_corridor_with_obstacles(
        self,
        corridor_mask: np.ndarray,
        frame: np.ndarray,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        height, width = corridor_mask.shape[:2]
        corridor_area = np.sum(corridor_mask > 0)
        if corridor_area == 0:
            return corridor_mask

        bottom_start = int(height * 0.5)
        bottom_corridor = corridor_mask[bottom_start:, :]
        if np.sum(bottom_corridor > 0) == 0:
            return corridor_mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        high_threshold = max(
            self._obstacle_edge_threshold * 2, 100
        )
        edges = cv2.Canny(
            blurred,
            high_threshold // 2,
            high_threshold,
        )

        bottom_edges = edges[bottom_start:, :]
        bottom_road = road_mask[bottom_start:, :]

        corridor_edges = cv2.bitwise_and(
            bottom_edges, bottom_corridor
        )

        kernel = np.ones((5, 5), dtype=np.uint8)
        dilated_edges = cv2.dilate(
            corridor_edges, kernel, iterations=3
        )

        obstacle_candidates = cv2.bitwise_and(
            dilated_edges, bottom_road
        )

        contours, _ = cv2.findContours(
            obstacle_candidates,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = max(
            10,
            int(
                height * width
                * self._obstacle_min_area_ratio
                * 2
            ),
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue
            pad_x = max(4, w // 3)
            pad_y = max(4, h // 3)
            x1 = max(0, x - pad_x)
            y1 = max(
                bottom_start,
                y + bottom_start - pad_y,
            )
            x2 = min(width, x + w + pad_x)
            y2 = min(
                height,
                y + bottom_start + h + pad_y,
            )

            obstacle_region_mask = np.zeros_like(corridor_mask)
            cv2.rectangle(
                obstacle_region_mask,
                (x1, y1),
                (x2, y2),
                255,
                -1,
            )
            overlap = cv2.bitwise_and(
                obstacle_region_mask, corridor_mask
            )
            overlap_area = np.sum(overlap > 0)
            if overlap_area > 0:
                corridor_mask = cv2.bitwise_and(
                    corridor_mask,
                    cv2.bitwise_not(overlap),
                )

        corridor_mask = cv2.morphologyEx(
            corridor_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return corridor_mask

    # 分析中心线的方向和弯道风险
    def _analyze_direction(
        self,
        centerline: List[Tuple[int, int]],
        height: int,
        width: int,
    ) -> Tuple[str, float]:
        if len(centerline) < 5:
            return "unknown", 0.5

        recent = centerline[-min(len(centerline), 10):]
        if len(recent) < 2:
            return "unknown", 0.5

        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2:]

        if not first_half or not second_half:
            return "unknown", 0.5

        first_cx = np.mean([p[0] for p in first_half])
        second_cx = np.mean([p[0] for p in second_half])

        center_x = width / 2.0
        shift = second_cx - first_cx

        direction = "straight"
        direction_risk = 0.1

        if abs(shift) > width * 0.02:
            if shift > 0:
                direction = "right"
            else:
                direction = "left"
            direction_risk = min(
                1.0, abs(shift) / (width * 0.1)
            )

        centerline_xs = [p[0] for p in centerline]
        x_range = max(centerline_xs) - min(centerline_xs)
        normalized_shift = abs(second_cx - center_x) / (width / 2.0)
        direction_risk = max(
            direction_risk,
            min(1.0, normalized_shift * 0.5),
        )

        return direction, float(direction_risk)

    # 基于走廊与障碍物重叠估算风险
    def _estimate_obstacle_risk(
        self,
        corridor_mask: np.ndarray,
        frame: np.ndarray,
        height: int,
        width: int,
    ) -> float:
        corridor_area = np.sum(corridor_mask > 0)
        if corridor_area == 0:
            return 1.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(sobel_x, sobel_y)

        corridor_gradient = gradient_mag[corridor_mask > 0]
        if len(corridor_gradient) == 0:
            return 0.1

        high_gradient_ratio = np.mean(
            corridor_gradient > self._obstacle_edge_threshold
        )

        bottom_region = corridor_mask[
            int(height * 0.6):, :
        ]
        if np.sum(bottom_region > 0) > 0:
            bottom_gray = gray[int(height * 0.6):, :]
            bottom_gradients = gradient_mag[
                int(height * 0.6):, :
            ][bottom_region > 0]
            if len(bottom_gradients) > 0:
                bottom_risk = np.mean(
                    bottom_gradients > self._obstacle_edge_threshold
                )
                high_gradient_ratio = max(
                    high_gradient_ratio, bottom_risk
                )

        risk = min(1.0, high_gradient_ratio * 3.0)
        return float(risk)

    # 基于走廊宽度变化估算风险
    def _estimate_width_risk(
        self,
        corridor_mask: np.ndarray,
        road_mask: np.ndarray,
    ) -> float:
        height, width = corridor_mask.shape[:2]

        road_area = np.sum(road_mask > 0)
        if road_area == 0:
            return 1.0

        corridor_area = np.sum(corridor_mask > 0)
        if corridor_area == 0:
            return 1.0

        ratio = corridor_area / road_area
        if ratio > 0.6:
            return 0.1
        elif ratio > 0.4:
            return 0.3
        elif ratio > 0.2:
            return 0.6
        else:
            return 0.9

    # 综合计算置信度（1.0=安全，0.0=危险）
    def _compute_confidence(
        self,
        obstacle_risk: float,
        direction_risk: float,
        width_risk: float,
    ) -> float:
        weighted_risk = (
            obstacle_risk * self._obstacle_risk_weight
            + direction_risk * self._direction_risk_weight
            + width_risk * self._width_risk_weight
        )
        confidence = 1.0 - weighted_risk
        return max(0.0, min(1.0, confidence))

    # 对置信度进行时序平滑，防止逐帧跳变
    def _temporal_smooth_confidence(
        self,
        raw_confidence: float,
    ) -> float:
        if not self._history_buffer:
            return raw_confidence

        recent_confidences = [
            h["confidence"]
            for h in self._history_buffer
        ]

        weights = [
            self._confidence_decay ** i
            for i in range(len(recent_confidences))
        ]
        weighted_sum = sum(
            c * w
            for c, w in zip(
                reversed(recent_confidences), weights
            )
        )
        weight_total = sum(weights)
        historical_avg = (
            weighted_sum / weight_total
            if weight_total > 0
            else raw_confidence
        )

        smoothed = (
            0.6 * raw_confidence + 0.4 * historical_avg
        )
        return max(0.0, min(1.0, smoothed))

    # 对中心线进行等距加密，输出固定数量点
    def _densify_centerline(
        self,
        centerline: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if len(centerline) < 2:
            return centerline

        if len(centerline) >= self._max_centerline_points:
            step = max(
                1,
                len(centerline)
                // self._max_centerline_points,
            )
            return centerline[::step]

        target = min(
            self._max_centerline_points,
            max(len(centerline), 20),
        )
        dense_line: List[Tuple[int, int]] = []
        for i in range(len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i + 1]
            if i == len(centerline) - 2:
                steps = max(
                    1,
                    target // max(1, len(centerline) - 1),
                )
            else:
                steps = max(
                    1,
                    target // max(1, len(centerline) - 1),
                )
            for s in range(steps):
                t = s / steps
                nx = int(p1[0] + t * (p2[0] - p1[0]))
                ny = int(p1[1] + t * (p2[1] - p1[1]))
                dense_line.append((nx, ny))

        return dense_line[: self._max_centerline_points]

    # 将当前帧结果存入历史缓冲区
    def _update_history(
        self,
        corridor_mask: np.ndarray,
        polygon: List[Tuple[int, int]],
        centerline: List[Tuple[int, int]],
        confidence: float,
    ) -> None:
        self._history_buffer.append({
            "corridor_mask": corridor_mask.copy(),
            "polygon": list(polygon),
            "centerline": list(centerline),
            "confidence": confidence,
        })