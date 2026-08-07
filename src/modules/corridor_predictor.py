"""
文件名: corridor_predictor.py
用途: 预测视频图像空间中的自车行驶走廊，检测碰撞风险，给出方向建议
作者: 温涵清
创建日期: 2026-07-28
最后修改日期: 2026-08-07
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
from src.interface.schemas import DetectedObject
from src.interface.schemas import UnknownRegion


INVALID_INPUT_ERROR = -1
MODEL_LOAD_ERROR = -2
INFERENCE_ERROR = -3
SUCCESS = 0

# 历史结果最大缓存帧数
MAX_HISTORY_SIZE = 10


class CorridorPredictor(CorridorPredictorInterface):
    """预测自车行驶走廊，估算几何风险，建议行驶方向"""

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
        self._corridor_width_ratio = float(
            config.get("corridor_width_ratio", 0.5)
        )
        self._corridor_top_ratio = float(
            config.get("corridor_top_ratio", 0.8)
        )
        self._geometry_risk_weight = float(
            config.get("geometry_risk_weight", 0.4)
        )
        self._direction_risk_weight = float(
            config.get("direction_risk_weight", 0.3)
        )
        self._width_risk_weight = float(
            config.get("width_risk_weight", 0.3)
        )
        self._history_buffer: Deque[dict] = deque(
            maxlen=MAX_HISTORY_SIZE
        )
        
        # 新增：预测时间窗口（秒）
        self._prediction_horizon_seconds = float(
            config.get("prediction_horizon_seconds", 3.0)
        )
        # 新增：自车速度（米/秒），用于计算时间
        self._ego_speed_mps = float(
            config.get("ego_speed_mps", 13.9)
        )
        
        # 新增：几何投影参数（与 distance_estimator 保持一致）
        self._focal_length_px = float(
            config.get("focal_length_px", 1000.0)
        )
        self._camera_height_m = float(
            config.get("camera_height_m", 1.50)
        )
        self._horizon_ratio = float(
            config.get("horizon_ratio", 0.50)
        )
        self._min_distance_m = float(
            config.get("min_distance_m", 0.3)
        )
        self._max_distance_m = float(
            config.get("max_distance_m", 80.0)
        )
        
        # 新增：缓存最近一帧的障碍物避让结果和预测信息
        self._last_obstacle_cutout_mask: Optional[np.ndarray] = None
        self._last_prediction_markers: List[dict] = []

    # 清空所有跨帧历史状态
    def reset(self) -> None:
        self._history_buffer.clear()
        self._last_obstacle_cutout_mask = None
        self._last_prediction_markers = []
        
    # 获取最近一帧的障碍物避让结果
    def get_obstacle_avoidance_result(self) -> Optional[np.ndarray]:
        return self._last_obstacle_cutout_mask
        
    # 获取最近一帧的预测信息（时间和距离标记）
    def get_prediction_info(self) -> List[dict]:
        return self._last_prediction_markers

    # 预测当前帧的自车行驶走廊
    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
        known_objects: Optional[List[DetectedObject]] = None,
        unknown_regions: Optional[List[UnknownRegion]] = None,
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
            
            # 新增：应用障碍物避让逻辑，缓存结果
            self._last_obstacle_cutout_mask = (
                self._apply_obstacle_avoidance(
                    corridor_mask, known_objects, unknown_regions
                )
            )
            
            # 新增：计算预测信息，缓存结果
            frame_height = frame.shape[0]
            self._last_prediction_markers = (
                self._compute_prediction_markers(
                    centerline, frame_height
                )
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

    # 核心算法：从道路掩码拟合直线边缘，构建梯形走廊并估算风险
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

        trapezoid_result = self._build_trapezoid_corridor(
            left_boundary,
            right_boundary,
            valid_rows,
            height,
            width,
        )

        if trapezoid_result is None:
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            self._update_history(
                empty_mask, [], [], 0.0
            )
            return empty_mask, [], [], 0.0

        _, _, polygon, centerline = trapezoid_result

        if len(polygon) < 3 or len(centerline) < 2:
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            self._update_history(
                empty_mask, [], [], 0.0
            )
            return empty_mask, [], [], 0.0

        corridor_mask = self._build_corridor_mask(
            polygon, height, width
        )

        direction, direction_risk = self._analyze_direction(
            centerline, height, width
        )

        geometry_risk = self._estimate_geometry_risk(
            valid_rows, height, width
        )

        width_risk = self._estimate_width_risk(
            corridor_mask, road_mask
        )

        raw_confidence = self._compute_confidence(
            geometry_risk, direction_risk, width_risk
        )

        smoothed_confidence = self._temporal_smooth_confidence(
            raw_confidence
        )

        self._update_history(
            corridor_mask,
            polygon,
            centerline,
            smoothed_confidence,
        )

        return (
            corridor_mask,
            polygon,
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

    # 对道路边界进行直线拟合，构建边缘平行于道路的等比例缩小梯形
    def _build_trapezoid_corridor(
        self,
        left_boundary: dict,
        right_boundary: dict,
        valid_rows: List[int],
        height: int,
        width: int,
    ) -> Optional[Tuple[dict, dict, List[Tuple[int, int]],
                        List[Tuple[int, int]]]]:
        sorted_rows = sorted(valid_rows)

        # 收集边界点用于直线拟合
        left_ys: List[float] = []
        left_xs: List[float] = []
        right_ys: List[float] = []
        right_xs: List[float] = []
        for y in sorted_rows:
            lx = left_boundary.get(y)
            rx = right_boundary.get(y)
            if lx is not None and rx is not None:
                left_ys.append(float(y))
                left_xs.append(float(lx))
                right_ys.append(float(y))
                right_xs.append(float(rx))

        if len(left_ys) < 2 or len(right_ys) < 2:
            return None

        # 线性拟合道路边缘：x = a * y + b
        left_coeffs = np.polyfit(left_ys, left_xs, 1)
        right_coeffs = np.polyfit(right_ys, right_xs, 1)
        left_a = float(left_coeffs[0])
        left_b = float(left_coeffs[1])
        right_a = float(right_coeffs[0])
        right_b = float(right_coeffs[1])

        # 计算消失点 y（左右边缘直线的交点）
        if abs(left_a - right_a) > 1e-6:
            vp_y = (right_b - left_b) / (left_a - right_a)
        else:
            vp_y = float(sorted_rows[0])

        road_bottom_y = float(sorted_rows[-1])
        road_top_y = float(sorted_rows[0])

        # 车头正中间的横向位置（图像中心 x）
        ego_center_x = width / 2.0

        # 检查拟合是否合理：消失点应在道路上方
        # 若拟合结果不合理（模型输出噪声大），使用对称梯形回退
        if vp_y >= road_bottom_y:
            # 回退：假设消失点在画面上方 1/3 处，横向对齐车头中心
            vp_y = height * (1.0 / 3.0)
            vp_x = ego_center_x
            actual_bottom_y = int(road_bottom_y)
            actual_bl = float(
                left_boundary.get(actual_bottom_y, 0)
            )
            actual_br = float(
                right_boundary.get(
                    actual_bottom_y, width - 1
                )
            )
            actual_width = actual_br - actual_bl
            if actual_width < self._min_road_width:
                return None
            # 对称边缘：从车头正中间向两侧等距扩展，斜边收敛到消失点
            half_w_raw = actual_width / 2.0
            dy = vp_y - road_bottom_y
            if abs(dy) < 1.0:
                dy = -1.0
            left_a = (vp_x - (ego_center_x - half_w_raw)) / dy
            right_a = (vp_x - (ego_center_x + half_w_raw)) / dy
            left_b = (
                ego_center_x - half_w_raw
            ) - left_a * road_bottom_y
            right_b = (
                ego_center_x + half_w_raw
            ) - right_a * road_bottom_y
        else:
            # 正常拟合：把道路中心平移到车头正中间，保持左右斜边斜率
            fit_center_at_bottom = (
                (left_a * road_bottom_y + left_b)
                + (right_a * road_bottom_y + right_b)
            ) / 2.0
            offset_x = ego_center_x - fit_center_at_bottom
            left_b += offset_x
            right_b += offset_x
            # 重新计算消失点 x 位置（对齐到车头正中间）
            vp_x = left_a * vp_y + left_b

        # 梯形底边在道路 mask 实际检测到的最底部（车头前的道路边缘）
        trap_bottom_y = int(road_bottom_y)
        # 梯形顶边：从底边向消失点方向延伸 corridor_top_ratio
        trap_top_y = int(
            trap_bottom_y
            + (vp_y - trap_bottom_y) * self._corridor_top_ratio
        )
        # 不超过道路实际可见范围
        trap_top_y = max(trap_top_y, int(road_top_y))
        # 保证梯形有足够高度
        trap_top_y = min(trap_top_y, trap_bottom_y - 10)

        # 底边处道路宽度（按平移后的道路边缘计算）
        road_bottom_left = left_a * trap_bottom_y + left_b
        road_bottom_right = right_a * trap_bottom_y + right_b
        road_bottom_width = (
            road_bottom_right - road_bottom_left
        )

        if road_bottom_width < self._min_road_width:
            return None

        # 梯形宽度为道路宽度的 corridor_width_ratio，以车头正中间为基准
        scale = self._corridor_width_ratio
        trap_bottom_left = (
            ego_center_x - (road_bottom_width * scale) / 2.0
        )
        trap_bottom_right = (
            ego_center_x + (road_bottom_width * scale) / 2.0
        )

        # 梯形左边缘平行于道路左边缘（斜率 left_a），起点在底边左下角
        trap_left_b = trap_bottom_left - left_a * trap_bottom_y
        # 梯形右边缘平行于道路右边缘（斜率 right_a），起点在底边右下角
        trap_right_b = (
            trap_bottom_right - right_a * trap_bottom_y
        )

        # 顶边处梯形角点
        trap_top_left = left_a * trap_top_y + trap_left_b
        trap_top_right = right_a * trap_top_y + trap_right_b

        # 限制到图像范围内
        def _clamp_x(x: float) -> int:
            return int(max(0, min(width - 1, x)))

        polygon = [
            (_clamp_x(trap_bottom_left), trap_bottom_y),
            (_clamp_x(trap_bottom_right), trap_bottom_y),
            (_clamp_x(trap_top_right), trap_top_y),
            (_clamp_x(trap_top_left), trap_top_y),
        ]

        # 逐行构建走廊边界
        corridor_left: dict = {}
        corridor_right: dict = {}
        for y in range(trap_top_y, trap_bottom_y + 1):
            cl = left_a * y + trap_left_b
            cr = right_a * y + trap_right_b
            corridor_left[y] = _clamp_x(cl)
            corridor_right[y] = _clamp_x(cr)

        # 从底到顶构建中心线
        centerline: List[Tuple[int, int]] = []
        num_points = min(50, trap_bottom_y - trap_top_y + 1)
        if num_points >= 2:
            for i in range(num_points):
                t = i / (num_points - 1)
                y = int(
                    trap_bottom_y
                    + (trap_top_y - trap_bottom_y) * t
                )
                cx = (
                    left_a * y + trap_left_b
                    + right_a * y + trap_right_b
                ) / 2.0
                centerline.append((_clamp_x(cx), y))

        return (
            corridor_left,
            corridor_right,
            polygon,
            centerline,
        )

    # 构建走廊多边形（底部→顶部→返回底部）
    def _build_corridor_polygon(
        self,
        corridor_boundaries: Tuple[dict, dict],
        valid_rows: List[int],
    ) -> List[Tuple[int, int]]:
        corridor_left, corridor_right = corridor_boundaries

        if not valid_rows:
            return []

        sorted_rows = sorted(valid_rows)
        left_points: List[Tuple[int, int]] = []
        right_points: List[Tuple[int, int]] = []

        for y in sorted_rows:
            lx = corridor_left.get(y)
            rx = corridor_right.get(y)
            if lx is not None:
                left_points.append((lx, y))
            if rx is not None:
                right_points.append((rx, y))

        if not left_points or not right_points:
            return []

        polygon: List[Tuple[int, int]] = []
        for p in reversed(left_points):
            polygon.append(p)
        for p in right_points:
            polygon.append(p)

        return polygon

    # 对走廊掩码进行形态学平滑，获得平滑边缘
    def _smooth_corridor_edges(
        self,
        corridor_mask: np.ndarray,
    ) -> np.ndarray:
        if np.sum(corridor_mask > 0) == 0:
            return corridor_mask

        kernel = np.ones((5, 5), dtype=np.uint8)
        smoothed = cv2.morphologyEx(
            corridor_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
        smoothed = cv2.morphologyEx(
            smoothed,
            cv2.MORPH_OPEN,
            kernel,
        )

        return smoothed

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
        normalized_shift = (
            abs(second_cx - center_x) / (width / 2.0)
        )
        direction_risk = max(
            direction_risk,
            min(1.0, normalized_shift * 0.5),
        )

        return direction, float(direction_risk)

    # 基于道路几何特征估算风险（连续性、宽度稳定性）
    def _estimate_geometry_risk(
        self,
        valid_rows: List[int],
        height: int,
        width: int,
    ) -> float:
        if not valid_rows:
            return 1.0

        row_coverage = len(valid_rows) / max(
            1,
            height // self._boundary_sample_step,
        )
        coverage_risk = 1.0 - min(1.0, row_coverage * 1.5)

        if len(valid_rows) >= 2:
            diffs = np.diff(valid_rows)
            gap_count = np.sum(diffs > self._boundary_sample_step * 2)
            gap_risk = min(1.0, gap_count / max(1, len(diffs)))
        else:
            gap_risk = 0.5

        risk = 0.6 * coverage_risk + 0.4 * gap_risk
        return float(min(1.0, max(0.0, risk)))

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
        if ratio > 0.5:
            return 0.1
        elif ratio > 0.3:
            return 0.3
        elif ratio > 0.15:
            return 0.6
        else:
            return 0.9

    # 综合计算置信度（1.0=安全，0.0=危险）
    def _compute_confidence(
        self,
        geometry_risk: float,
        direction_risk: float,
        width_risk: float,
    ) -> float:
        weighted_risk = (
            geometry_risk * self._geometry_risk_weight
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

    # 应用障碍物避让逻辑，从走廊中扣除障碍物区域
    def _apply_obstacle_avoidance(
        self,
        corridor_mask: np.ndarray,
        known_objects: Optional[List[DetectedObject]],
        unknown_regions: Optional[List[UnknownRegion]],
    ) -> np.ndarray:
        if corridor_mask is None or corridor_mask.size == 0:
            return corridor_mask
            
        height, width = corridor_mask.shape[:2]
        cutout_mask = corridor_mask.copy()
        
        obstacles_to_cut = []
        
        if known_objects:
            for obj in known_objects:
                if obj.bbox:
                    obstacles_to_cut.append(obj.bbox)
                    
        if unknown_regions:
            for region in unknown_regions:
                if region.bbox:
                    obstacles_to_cut.append(region.bbox)
                    
        for bbox in obstacles_to_cut:
            x1, y1, x2, y2 = bbox
            # 确保坐标在图像范围内
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            
            if x2 > x1 and y2 > y1:
                cutout_mask[y1:y2, x1:x2] = 0
                
        return cutout_mask

    # 计算预测信息（基于几何投影的真实距离和时间）
    def _compute_prediction_markers(
        self,
        centerline: List[Tuple[int, int]],
        frame_height: int,
    ) -> List[dict]:
        if not centerline or len(centerline) < 2:
            return []
        
        markers = []
        total_points = len(centerline)
        
        # 计算地平线位置
        horizon_y = frame_height * self._horizon_ratio
        
        # 使用几何投影公式计算每个点的真实距离
        # 与 distance_estimator._estimate_ground_distance 保持一致
        # distance = focal_length * camera_height / vertical_offset
        def estimate_distance_at_y(y: int) -> Optional[float]:
            vertical_offset = float(y) - horizon_y
            if vertical_offset <= 0.0:
                return None
            if self._focal_length_px <= 0.0:
                return None
            if self._camera_height_m <= 0.0:
                return None
            distance = (
                self._focal_length_px
                * self._camera_height_m
                / vertical_offset
            )
            if not (self._min_distance_m <= distance <= self._max_distance_m):
                return None
            return float(distance)
        
        # 沿路径均匀采样几个标记点（例如 5 个点）
        num_markers = min(5, total_points)
        for i in range(num_markers):
            t = i / (num_markers - 1) if num_markers > 1 else 0
            point_idx = int(t * (total_points - 1))
            point = centerline[point_idx]
            
            # 使用几何投影计算真实距离
            real_distance = estimate_distance_at_y(point[1])
            
            if real_distance is None:
                # 距离无效时，跳过此点
                continue
            
            # 估算时间（秒）= 距离 / 自车速度
            if self._ego_speed_mps > 0:
                time_seconds = real_distance / self._ego_speed_mps
            else:
                time_seconds = 0.0
            
            markers.append({
                "x": point[0],
                "y": point[1],
                "time_s": round(time_seconds, 2),
                "distance_m": round(real_distance, 2),
            })
            
        return markers

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