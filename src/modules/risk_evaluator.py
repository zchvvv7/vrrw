"""
文件名: risk_evaluator.py
用途: 基于视频距离、行驶走廊和连续帧输出障碍物风险
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-08-04
"""

from collections import deque
from time import perf_counter
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from src.interface.schemas import CorridorPredictionResult
from src.interface.schemas import DetectedObject
from src.interface.schemas import ObstacleRisk
from src.interface.schemas import RiskEvaluationResult
from src.interface.schemas import SystemStatus
from src.interface.schemas import UnknownRegion


INVALID_INPUT_ERROR = -1
INFERENCE_ERROR = -3
SUCCESS = 0

RISK_ORDER = {
    "safe": 0,
    "notice": 1,
    "warning": 2,
    "danger": 3,
}


class RiskEvaluator:
    """基于视频连续帧评估障碍物空间冲突和风险等级"""

    # 初始化风险阈值和视频帧间状态
    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        self._enabled = bool(config.get("enabled", False))
        self._model_version = str(
            config.get("model_version", "video-risk-v1.0")
        )
        self._intersection_ratio = float(
            config.get("intersection_ratio", 0.15)
        )
        self._near_margin_ratio = float(
            config.get("near_margin_ratio", 0.025)
        )
        self._footprint_height_ratio = float(
            config.get("footprint_height_ratio", 0.25)
        )
        self._max_corridor_lateral_ratio = float(
            config.get("max_corridor_lateral_ratio", 0.65)
        )
        self._min_known_confidence = float(
            config.get("min_known_confidence", 0.45)
        )
        self._max_known_bbox_area_ratio = float(
            config.get("max_known_bbox_area_ratio", 0.40)
        )
        self._max_known_bbox_border_count = int(
            config.get("max_known_bbox_border_count", 2)
        )
        self._notice_distance_m = float(
            config.get("notice_distance_m", 30.0)
        )
        self._warning_distance_m = float(
            config.get("warning_distance_m", 15.0)
        )
        self._danger_distance_m = float(
            config.get("danger_distance_m", 6.0)
        )
        self._notice_ttc_s = float(
            config.get("notice_ttc_s", 5.0)
        )
        self._warning_ttc_s = float(
            config.get("warning_ttc_s", 3.0)
        )
        self._danger_ttc_s = float(
            config.get("danger_ttc_s", 1.5)
        )
        self._track_iou_threshold = float(
            config.get("track_iou_threshold", 0.30)
        )
        self._history_size = int(
            config.get("history_size", 15)
        )
        self._confirm_frames = int(
            config.get("confirm_frames", 3)
        )
        self._min_ttc_samples = int(
            config.get("min_ttc_samples", 8)
        )
        self._min_ttc_observation_s = float(
            config.get("min_ttc_observation_s", 0.25)
        )
        self._min_ttc_r_squared = float(
            config.get("min_ttc_r_squared", 0.80)
        )
        self._min_closing_observations_ratio = float(
            config.get("min_closing_observations_ratio", 0.75)
        )
        self._min_corridor_confidence = float(
            config.get("min_corridor_confidence", 0.45)
        )
        self._min_closing_speed_mps = float(
            config.get("min_closing_speed_mps", 0.5)
        )
        self._max_closing_speed_mps = float(
            config.get("max_closing_speed_mps", 25.0)
        )
        self._max_track_gap_s = float(
            config.get("max_track_gap_s", 0.5)
        )
        self._validate_configuration()
        self._tracks: Dict[str, dict] = {}
        self._next_track_index = 1

    # 清空跨视频保存的目标和风险状态
    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_index = 1

    # 计算当前视频帧的空间冲突、TTC和风险等级
    def evaluate(
        self,
        frame_id: int,
        fps: float,
        corridor_result: CorridorPredictionResult,
        known_objects: List[DetectedObject],
        unknown_regions: List[UnknownRegion],
        system_status: str = SystemStatus.NORMAL,
    ) -> RiskEvaluationResult:
        start_time = perf_counter()
        if not self._enabled:
            return self._build_disabled_result(system_status)

        input_error = self._validate_input(
            frame_id=frame_id,
            fps=fps,
            corridor_result=corridor_result,
            known_objects=known_objects,
            unknown_regions=unknown_regions,
        )
        if input_error is not None:
            return self._build_unavailable_result(
                start_time=start_time,
                error_code=INVALID_INPUT_ERROR,
                error_message=input_error,
                system_status=SystemStatus.UNAVAILABLE,
            )

        if system_status == SystemStatus.UNAVAILABLE:
            return self._build_unavailable_result(
                start_time=start_time,
                error_code=SUCCESS,
                error_message="Perception system is unavailable.",
                system_status=system_status,
            )

        if not self._is_corridor_usable(corridor_result):
            return self._build_unavailable_result(
                start_time=start_time,
                error_code=SUCCESS,
                error_message="Driving corridor is unavailable.",
                system_status=SystemStatus.DEGRADED,
            )

        try:
            obstacle_risks = self._evaluate_obstacles(
                frame_id=frame_id,
                fps=fps,
                corridor_mask=corridor_result.corridor_mask,
                known_objects=known_objects,
                unknown_regions=unknown_regions,
            )
            raw_level, raw_reason = self._select_frame_risk(
                obstacle_risks
            )
            inference_time_ms = (
                perf_counter() - start_time
            ) * 1000.0
            return RiskEvaluationResult(
                risk_level=raw_level,
                major_reason=raw_reason,
                obstacle_risks=obstacle_risks,
                system_status=system_status,
                is_valid=True,
                inference_time_ms=inference_time_ms,
                error_code=SUCCESS,
                error_message="OK",
                model_version=self._model_version,
                is_enabled=True,
            )
        except Exception as error:
            return self._build_unavailable_result(
                start_time=start_time,
                error_code=INFERENCE_ERROR,
                error_message=(
                    f"{type(error).__name__}: {error}"
                ),
                system_status=SystemStatus.UNAVAILABLE,
            )

    # 校验配置阈值范围和大小关系
    def _validate_configuration(self) -> None:
        if not 0.0 <= self._intersection_ratio <= 1.0:
            raise ValueError(
                "intersection_ratio must be between 0 and 1."
            )
        if not 0.0 <= self._near_margin_ratio <= 0.5:
            raise ValueError(
                "near_margin_ratio must be between 0 and 0.5."
            )
        if not 0.0 < self._footprint_height_ratio <= 1.0:
            raise ValueError(
                "footprint_height_ratio must be in (0, 1]."
            )
        if not 0.0 < self._max_corridor_lateral_ratio <= 1.0:
            raise ValueError(
                "max_corridor_lateral_ratio must be in (0, 1]."
            )
        if not 0.0 <= self._min_known_confidence <= 1.0:
            raise ValueError(
                "min_known_confidence must be between 0 and 1."
            )
        if not 0.0 < self._max_known_bbox_area_ratio <= 1.0:
            raise ValueError(
                "max_known_bbox_area_ratio must be in (0, 1]."
            )
        if not 0 <= self._max_known_bbox_border_count <= 4:
            raise ValueError(
                "max_known_bbox_border_count must be between 0 and 4."
            )
        if not (
            0.0 < self._danger_distance_m
            <= self._warning_distance_m
            <= self._notice_distance_m
        ):
            raise ValueError(
                "Distance thresholds must increase from danger "
                "to notice."
            )
        if not (
            0.0 < self._danger_ttc_s
            <= self._warning_ttc_s
            <= self._notice_ttc_s
        ):
            raise ValueError(
                "TTC thresholds must increase from danger to notice."
            )
        if self._history_size < 3:
            raise ValueError("history_size must be at least 3.")
        if self._confirm_frames < 1:
            raise ValueError("confirm_frames must be positive.")
        if not 3 <= self._min_ttc_samples <= self._history_size:
            raise ValueError(
                "min_ttc_samples must be between 3 and history_size."
            )
        if self._min_ttc_observation_s <= 0.0:
            raise ValueError(
                "min_ttc_observation_s must be positive."
            )
        if not 0.0 <= self._min_ttc_r_squared <= 1.0:
            raise ValueError(
                "min_ttc_r_squared must be between 0 and 1."
            )
        if not (
            0.5
            <= self._min_closing_observations_ratio
            <= 1.0
        ):
            raise ValueError(
                "min_closing_observations_ratio must be "
                "between 0.5 and 1."
            )
        if (
            self._max_closing_speed_mps
            <= self._min_closing_speed_mps
        ):
            raise ValueError(
                "max_closing_speed_mps must exceed "
                "min_closing_speed_mps."
            )
        if self._max_track_gap_s <= 0.0:
            raise ValueError("max_track_gap_s must be positive.")

    # 校验视频帧风险评估输入
    def _validate_input(
        self,
        frame_id: int,
        fps: float,
        corridor_result: CorridorPredictionResult,
        known_objects: List[DetectedObject],
        unknown_regions: List[UnknownRegion],
    ) -> Optional[str]:
        if frame_id < 0:
            return "frame_id cannot be negative."
        if not np.isfinite(fps) or fps <= 0.0:
            return "Video FPS must be positive."
        if not isinstance(
            corridor_result,
            CorridorPredictionResult,
        ):
            return "corridor_result has an invalid type."
        if not isinstance(known_objects, list):
            return "known_objects must be a list."
        if not isinstance(unknown_regions, list):
            return "unknown_regions must be a list."
        return None

    # 判断当前走廊是否足以支持空间冲突计算
    def _is_corridor_usable(
        self,
        corridor_result: CorridorPredictionResult,
    ) -> bool:
        corridor_mask = corridor_result.corridor_mask
        if not corridor_result.is_successful:
            return False
        if not corridor_result.is_enabled:
            return False
        if not isinstance(corridor_mask, np.ndarray):
            return False
        if corridor_mask.ndim != 2 or corridor_mask.size == 0:
            return False
        if not np.any(corridor_mask > 0):
            return False
        return (
            corridor_result.confidence
            >= self._min_corridor_confidence
        )

    # 构造关闭状态的风险结果
    def _build_disabled_result(
        self,
        system_status: str,
    ) -> RiskEvaluationResult:
        return RiskEvaluationResult(
            risk_level="safe",
            major_reason="risk_evaluator_disabled",
            obstacle_risks=[],
            system_status=system_status,
            is_valid=False,
            inference_time_ms=0.0,
            error_code=SUCCESS,
            error_message="Risk evaluation is disabled.",
            model_version=self._model_version,
            is_enabled=False,
        )

    # 构造不可用或失败状态的保守风险结果
    def _build_unavailable_result(
        self,
        start_time: float,
        error_code: int,
        error_message: str,
        system_status: str,
    ) -> RiskEvaluationResult:
        inference_time_ms = (
            perf_counter() - start_time
        ) * 1000.0
        return RiskEvaluationResult(
            risk_level="notice",
            major_reason="risk_evaluation_unavailable",
            obstacle_risks=[],
            system_status=system_status,
            is_valid=False,
            inference_time_ms=inference_time_ms,
            error_code=error_code,
            error_message=error_message,
            model_version=self._model_version,
            is_enabled=True,
        )

    # 将已知和未知障碍物统一转换为风险候选项
    def _evaluate_obstacles(
        self,
        frame_id: int,
        fps: float,
        corridor_mask: np.ndarray,
        known_objects: List[DetectedObject],
        unknown_regions: List[UnknownRegion],
    ) -> List[ObstacleRisk]:
        height, width = corridor_mask.shape
        near_mask = self._build_near_mask(corridor_mask)
        candidates: List[dict] = []

        for detected_object in known_objects:
            if not isinstance(detected_object, DetectedObject):
                continue
            if not self._is_plausible_known_object(
                detected_object=detected_object,
                image_shape=(height, width),
            ):
                continue
            occupancy_mask = self._build_bbox_footprint(
                bbox=detected_object.bbox,
                image_shape=(height, width),
            )
            candidates.append({
                "source": "known",
                "class_name": detected_object.class_name,
                "bbox": detected_object.bbox,
                "distance": detected_object.distance,
                "occupancy_mask": occupancy_mask,
            })

        for unknown_region in unknown_regions:
            if not isinstance(unknown_region, UnknownRegion):
                continue
            occupancy_mask = self._decode_mask_rle(
                unknown_region.mask_rle,
                (height, width),
            )
            if occupancy_mask is None:
                occupancy_mask = self._build_bbox_footprint(
                    bbox=unknown_region.bbox,
                    image_shape=(height, width),
                )
            candidates.append({
                "source": "unknown",
                "class_name": None,
                "bbox": unknown_region.bbox,
                "distance": unknown_region.distance,
                "occupancy_mask": occupancy_mask,
            })

        self._associate_tracks(candidates, frame_id, fps)
        obstacle_risks = []
        for candidate in candidates:
            overlap, relation = self._compute_spatial_relation(
                occupancy_mask=candidate["occupancy_mask"],
                corridor_mask=corridor_mask,
                near_mask=near_mask,
            )
            track = self._tracks[candidate["track_id"]]
            ttc = self._estimate_ttc(track, fps)
            risk_level, reason = self._classify_obstacle(
                source=candidate["source"],
                relation=relation,
                distance=candidate["distance"],
                ttc=ttc,
                stable_frames=track["stable_frames"],
            )
            obstacle_risks.append(
                ObstacleRisk(
                    object_id=candidate["track_id"],
                    source=candidate["source"],
                    class_name=candidate["class_name"],
                    bbox=tuple(candidate["bbox"]),
                    distance=candidate["distance"],
                    corridor_overlap=overlap,
                    spatial_relation=relation,
                    ttc=ttc,
                    risk_level=risk_level,
                    major_reason=reason,
                    stable_frames=track["stable_frames"],
                )
            )
        return obstacle_risks

    # 判断已知障碍物置信度和边界框是否足以进入风险计算
    def _is_plausible_known_object(
        self,
        detected_object: DetectedObject,
        image_shape: Tuple[int, int],
    ) -> bool:
        confidence = detected_object.confidence
        if not np.isfinite(confidence):
            return False
        if confidence < self._min_known_confidence:
            return False
        if len(detected_object.bbox) != 4:
            return False

        height, width = image_shape
        x1, y1, x2, y2 = [
            int(value) for value in detected_object.bbox
        ]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return False

        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = height * width
        area_ratio = bbox_area / float(image_area)
        if area_ratio > self._max_known_bbox_area_ratio:
            return False

        border_count = sum([
            x1 == 0,
            y1 == 0,
            x2 == width,
            y2 == height,
        ])
        return border_count <= self._max_known_bbox_border_count

    # 将边界框底部转换为近似路面占用掩码
    def _build_bbox_footprint(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros(image_shape, dtype=np.uint8)
        if len(bbox) != 4:
            return mask
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return mask
        box_height = y2 - y1
        footprint_height = max(
            1,
            int(round(box_height * self._footprint_height_ratio)),
        )
        footprint_y1 = max(y1, y2 - footprint_height)
        mask[footprint_y1:y2, x1:x2] = 255
        return mask

    # 解码未知障碍物的列优先游程掩码
    def _decode_mask_rle(
        self,
        mask_rle: Optional[dict],
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not isinstance(mask_rle, dict):
            return None
        size = mask_rle.get("size")
        counts = mask_rle.get("counts")
        if size != [image_shape[0], image_shape[1]]:
            return None
        if not isinstance(counts, list):
            return None
        total_pixels = image_shape[0] * image_shape[1]
        flattened = np.zeros(total_pixels, dtype=np.uint8)
        offset = 0
        foreground = False
        for raw_count in counts:
            count = int(raw_count)
            if count < 0 or offset + count > total_pixels:
                return None
            if foreground and count > 0:
                flattened[offset:offset + count] = 255
            offset += count
            foreground = not foreground
        if offset != total_pixels:
            return None
        return flattened.reshape(image_shape, order="F")

    # 生成走廊附近缓冲区域
    def _build_near_mask(
        self,
        corridor_mask: np.ndarray,
    ) -> np.ndarray:
        width = corridor_mask.shape[1]
        margin = max(
            1,
            int(round(width * self._near_margin_ratio)),
        )
        kernel_size = margin * 2 + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        return cv2.dilate(
            (corridor_mask > 0).astype(np.uint8) * 255,
            kernel,
            iterations=1,
        )

    # 计算障碍物占用区域与走廊的空间关系
    def _compute_spatial_relation(
        self,
        occupancy_mask: np.ndarray,
        corridor_mask: np.ndarray,
        near_mask: np.ndarray,
    ) -> Tuple[float, str]:
        obstacle_pixels = occupancy_mask > 0
        obstacle_area = int(np.count_nonzero(obstacle_pixels))
        if obstacle_area == 0:
            return 0.0, "outside"
        corridor_pixels = corridor_mask > 0
        intersection = int(
            np.count_nonzero(
                obstacle_pixels & corridor_pixels
            )
        )
        overlap = intersection / float(obstacle_area)
        contact_in_corridor = self._is_contact_in_mask(
            obstacle_pixels=obstacle_pixels,
            target_mask=corridor_pixels,
        )
        if (
            overlap >= self._intersection_ratio
            and contact_in_corridor
        ):
            return float(overlap), "intersecting"
        if np.any(obstacle_pixels & (near_mask > 0)):
            return float(overlap), "near"
        return float(overlap), "outside"

    # 判断障碍物底部接地点是否真正落入目标掩码
    def _is_contact_in_mask(
        self,
        obstacle_pixels: np.ndarray,
        target_mask: np.ndarray,
    ) -> bool:
        y_coordinates, x_coordinates = np.where(obstacle_pixels)
        if y_coordinates.size == 0:
            return False

        top_y = int(np.min(y_coordinates))
        bottom_y = int(np.max(y_coordinates))
        height = bottom_y - top_y + 1
        contact_band_height = max(1, int(round(height * 0.05)))
        contact_band_y = bottom_y - contact_band_height + 1
        band_x_coordinates = x_coordinates[
            y_coordinates >= contact_band_y
        ]
        if band_x_coordinates.size == 0:
            return False

        contact_x = int(round(float(np.median(band_x_coordinates))))
        contact_x = max(
            0,
            min(target_mask.shape[1] - 1, contact_x),
        )
        if not target_mask[bottom_y, contact_x]:
            return False

        corridor_x_coordinates = np.flatnonzero(
            target_mask[bottom_y]
        )
        if corridor_x_coordinates.size == 0:
            return False
        corridor_left = float(corridor_x_coordinates[0])
        corridor_right = float(corridor_x_coordinates[-1])
        corridor_center = (
            corridor_left + corridor_right
        ) / 2.0
        corridor_half_width = max(
            (corridor_right - corridor_left) / 2.0,
            0.5,
        )
        lateral_ratio = (
            abs(float(contact_x) - corridor_center)
            / corridor_half_width
        )
        return lateral_ratio <= self._max_corridor_lateral_ratio

    # 将当前障碍物与历史视频目标做轻量级关联
    def _associate_tracks(
        self,
        candidates: List[dict],
        frame_id: int,
        fps: float,
    ) -> None:
        self._prune_tracks(frame_id, fps)
        matched_tracks = set()
        for candidate in candidates:
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self._tracks.items():
                if track_id in matched_tracks:
                    continue
                if track["source"] != candidate["source"]:
                    continue
                if track["class_name"] != candidate["class_name"]:
                    continue
                iou = self._bbox_iou(
                    candidate["bbox"],
                    track["bbox"],
                )
                if (
                    iou >= self._track_iou_threshold
                    and iou > best_iou
                ):
                    best_track_id = track_id
                    best_iou = iou
            if best_track_id is None:
                best_track_id = self._create_track(
                    candidate,
                    frame_id,
                )
            else:
                self._update_track(
                    best_track_id,
                    candidate,
                    frame_id,
                    fps,
                )
            candidate["track_id"] = best_track_id
            matched_tracks.add(best_track_id)

    # 创建一个新的视频障碍物轨迹
    def _create_track(
        self,
        candidate: dict,
        frame_id: int,
    ) -> str:
        track_id = (
            f"{candidate['source']}-track-"
            f"{self._next_track_index:04d}"
        )
        self._next_track_index += 1
        distance_history: Deque[Tuple[int, float]] = deque(
            maxlen=self._history_size
        )
        distance = candidate["distance"]
        if self._is_valid_distance(distance):
            distance_history.append((frame_id, float(distance)))
        self._tracks[track_id] = {
            "source": candidate["source"],
            "class_name": candidate["class_name"],
            "bbox": tuple(candidate["bbox"]),
            "last_frame_id": frame_id,
            "stable_frames": 1,
            "distance_history": distance_history,
        }
        return track_id

    # 更新已经关联成功的视频障碍物轨迹
    def _update_track(
        self,
        track_id: str,
        candidate: dict,
        frame_id: int,
        fps: float,
    ) -> None:
        track = self._tracks[track_id]
        gap_s = (
            frame_id - track["last_frame_id"]
        ) / fps
        if gap_s <= self._max_track_gap_s:
            track["stable_frames"] += 1
        else:
            track["stable_frames"] = 1
            track["distance_history"].clear()
        track["bbox"] = tuple(candidate["bbox"])
        track["last_frame_id"] = frame_id
        distance = candidate["distance"]
        if self._is_valid_distance(distance):
            track["distance_history"].append(
                (frame_id, float(distance))
            )

    # 删除超过允许时间未再次出现的历史轨迹
    def _prune_tracks(
        self,
        frame_id: int,
        fps: float,
    ) -> None:
        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if (
                frame_id - track["last_frame_id"]
            ) / fps > self._max_track_gap_s
        ]
        for track_id in stale_track_ids:
            del self._tracks[track_id]

    # 根据距离历史线性拟合相对接近速度和TTC
    def _estimate_ttc(
        self,
        track: dict,
        fps: float,
    ) -> Optional[float]:
        history = list(track["distance_history"])
        if len(history) < self._min_ttc_samples:
            return None
        first_frame_id = history[0][0]
        times = np.asarray(
            [
                (frame_id - first_frame_id) / fps
                for frame_id, _distance in history
            ],
            dtype=np.float64,
        )
        distances = np.asarray(
            [distance for _frame_id, distance in history],
            dtype=np.float64,
        )
        if np.ptp(times) <= 0.0:
            return None
        observation_time = float(times[-1] - times[0])
        if observation_time < self._min_ttc_observation_s:
            return None

        distance_changes = np.diff(distances)
        closing_ratio = float(
            np.count_nonzero(distance_changes < 0.0)
            / distance_changes.size
        )
        if (
            closing_ratio
            < self._min_closing_observations_ratio
        ):
            return None

        slope, intercept = np.polyfit(times, distances, 1)
        slope = float(slope)
        fitted_distances = slope * times + float(intercept)
        residual_sum = float(
            np.sum((distances - fitted_distances) ** 2)
        )
        total_sum = float(
            np.sum((distances - np.mean(distances)) ** 2)
        )
        if total_sum <= np.finfo(np.float64).eps:
            return None
        r_squared = 1.0 - residual_sum / total_sum
        if r_squared < self._min_ttc_r_squared:
            return None

        closing_speed = -slope
        if not (
            self._min_closing_speed_mps
            <= closing_speed
            <= self._max_closing_speed_mps
        ):
            return None
        current_distance = float(distances[-1])
        ttc = current_distance / closing_speed
        if not np.isfinite(ttc) or ttc < 0.0:
            return None
        return float(ttc)

    # 根据空间关系、距离、TTC和连续性判定单目标风险
    def _classify_obstacle(
        self,
        source: str,
        relation: str,
        distance: Optional[float],
        ttc: Optional[float],
        stable_frames: int,
    ) -> Tuple[str, str]:
        if relation == "outside":
            return "safe", f"{source}_obstacle_outside_corridor"
        if stable_frames < self._confirm_frames:
            return "notice", f"{source}_obstacle_unconfirmed"

        danger_by_distance = (
            self._is_valid_distance(distance)
            and float(distance) <= self._danger_distance_m
        )
        danger_by_ttc = (
            self._is_valid_distance(distance)
            and ttc is not None
            and float(distance) <= self._warning_distance_m
            and ttc <= self._danger_ttc_s
        )
        warning_by_distance = (
            self._is_valid_distance(distance)
            and float(distance) <= self._warning_distance_m
        )
        warning_by_ttc = (
            self._is_valid_distance(distance)
            and ttc is not None
            and float(distance) <= self._notice_distance_m
            and ttc <= self._warning_ttc_s
        )

        if relation == "intersecting":
            if danger_by_distance:
                return "danger", (
                    f"{source}_obstacle_intersecting_"
                    "critical_distance"
                )
            if danger_by_ttc:
                return "danger", (
                    f"{source}_obstacle_intersecting_"
                    "critical_ttc"
                )
            if warning_by_distance:
                return "warning", (
                    f"{source}_obstacle_intersecting_"
                    "warning_distance"
                )
            if warning_by_ttc:
                return "warning", (
                    f"{source}_obstacle_intersecting_"
                    "warning_ttc"
                )
            if not self._is_valid_distance(distance):
                return "warning", (
                    f"{source}_obstacle_intersecting_"
                    "distance_unavailable"
                )
            return "notice", (
                f"{source}_obstacle_intersecting_far"
            )

        if danger_by_distance or warning_by_ttc:
            return "warning", f"{source}_obstacle_near_close"
        return "notice", f"{source}_obstacle_near_corridor"

    # 从所有障碍物中选择当前帧最高风险
    def _select_frame_risk(
        self,
        obstacle_risks: List[ObstacleRisk],
    ) -> Tuple[str, str]:
        if not obstacle_risks:
            return "safe", "no_obstacle"
        highest = max(
            obstacle_risks,
            key=lambda item: RISK_ORDER[item.risk_level],
        )
        if highest.risk_level == "safe":
            return "safe", "all_obstacles_outside_corridor"
        return highest.risk_level, highest.major_reason

    # 判断距离是否为有效正数
    def _is_valid_distance(
        self,
        distance: Optional[float],
    ) -> bool:
        if distance is None:
            return False
        return bool(np.isfinite(distance) and distance > 0.0)

    # 计算两个边界框的交并比
    def _bbox_iou(
        self,
        first_bbox: Tuple[int, int, int, int],
        second_bbox: Tuple[int, int, int, int],
    ) -> float:
        first_x1, first_y1, first_x2, first_y2 = first_bbox
        second_x1, second_y1, second_x2, second_y2 = second_bbox
        intersection_width = max(
            0,
            min(first_x2, second_x2)
            - max(first_x1, second_x1),
        )
        intersection_height = max(
            0,
            min(first_y2, second_y2)
            - max(first_y1, second_y1),
        )
        intersection_area = (
            intersection_width * intersection_height
        )
        first_area = max(
            0,
            (first_x2 - first_x1)
            * (first_y2 - first_y1),
        )
        second_area = max(
            0,
            (second_x2 - second_x1)
            * (second_y2 - second_y1),
        )
        union_area = first_area + second_area - intersection_area
        if union_area <= 0:
            return 0.0
        return float(intersection_area / union_area)
