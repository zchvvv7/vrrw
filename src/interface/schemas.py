"""
文件名: schemas.py
用途: 定义系统各模块之间传递的数据结构
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-08-03
"""

from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


@dataclass
class DetectedObject:
    """表示一个已知障碍物检测结果"""

    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    distance: Optional[float] = None


@dataclass
class KnownDetectionResult:
    """表示已知障碍物检测结果"""

    objects: List[DetectedObject]
    inference_time_ms: float
    error_code: int
    error_message: str
    model_version: str

    # 判断已知障碍物检测是否成功
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0


@dataclass
class DistanceEstimationResult:
    """表示仅针对已知障碍物的距离估计结果"""

    known_objects: List[DetectedObject]
    inference_time_ms: float
    error_code: int
    error_message: str
    method: str
    model_version: str
    is_enabled: bool

    # 判断距离估计是否成功或已按配置关闭
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0


@dataclass
class UnknownRegion:
    """表示一个未知异常区域"""

    bbox: Tuple[int, int, int, int]
    score: float
    distance: Optional[float] = None
    object_id: Optional[str] = None
    area: Optional[int] = None
    mask_rle: Optional[dict] = None


@dataclass
class ObstacleRisk:
    """表示单个障碍物与自车行驶走廊的风险关系"""

    object_id: str
    source: str
    class_name: Optional[str]
    bbox: Tuple[int, int, int, int]
    distance: Optional[float]
    corridor_overlap: float
    spatial_relation: str
    ttc: Optional[float]
    risk_level: str
    major_reason: str
    stable_frames: int


@dataclass
class RiskEvaluationResult:
    """表示视频当前帧的空间冲突与风险评估结果"""

    risk_level: str
    major_reason: str
    obstacle_risks: List[ObstacleRisk]
    system_status: str
    is_valid: bool
    inference_time_ms: float
    error_code: int
    error_message: str
    model_version: str
    is_enabled: bool

    # 判断风险评估是否执行成功或已按配置关闭
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0

    # 返回当前帧全部障碍物中的最小有效TTC
    @property
    def ttc(self) -> Optional[float]:
        valid_values = [
            item.ttc for item in self.obstacle_risks if item.ttc is not None
        ]
        if not valid_values:
            return None
        return float(min(valid_values))

    # 返回当前帧全部障碍物中的最大走廊交叠率
    @property
    def corridor_overlap(self) -> float:
        if not self.obstacle_risks:
            return 0.0
        return float(
            max(item.corridor_overlap for item in self.obstacle_risks)
        )


class SystemStatus:
    """表示系统状态"""

    NORMAL = "normal"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class RoadSegmentResult:
    """表示可行驶区域分割结果"""

    mask: np.ndarray
    boundary: Optional[np.ndarray]
    confidence_map: Optional[np.ndarray]
    global_confidence: float
    road_pixel_ratio: float
    error_code: int
    error_message: str
    inference_time_ms: float
    timestamp: Optional[float] = None
    system_status: str = SystemStatus.NORMAL
    quality_metrics: Optional[dict] = None

    # 判断道路分割是否成功
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0

    # 判断道路分割是否处于降级状态
    @property
    def is_degraded(self) -> bool:
        return self.system_status == SystemStatus.DEGRADED

    # 判断道路分割是否不可用
    @property
    def is_unavailable(self) -> bool:
        return self.system_status == SystemStatus.UNAVAILABLE

    # 将道路分割结果转换为字典
    def to_dict(self) -> dict:
        return {
            "mask": self.mask,
            "boundary": self.boundary,
            "confidence_map": self.confidence_map,
            "global_confidence": self.global_confidence,
            "road_pixel_ratio": self.road_pixel_ratio,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
            "system_status": self.system_status,
            "quality_metrics": self.quality_metrics,
        }


@dataclass
class CorridorPredictionResult:
    """表示视频当前帧中的自车行驶走廊预测结果"""

    corridor_mask: Optional[np.ndarray]
    polygon: List[Tuple[int, int]]
    centerline: List[Tuple[int, int]]
    confidence: float
    inference_time_ms: float
    error_code: int
    error_message: str
    method: str
    model_version: str
    is_enabled: bool

    # 判断行驶走廊预测是否成功或已按配置关闭
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0


@dataclass
class UnknownDetectionResult:
    """表示未知障碍物检测结果"""

    score_map: np.ndarray
    anomaly_mask: np.ndarray
    regions: List[UnknownRegion]
    inference_time_ms: float
    error_code: int
    error_message: str
    model_version: str

    # 判断未知障碍物检测是否成功
    @property
    def is_successful(self) -> bool:
        return self.error_code == 0


@dataclass
class FrameResult:
    """保存单帧处理结果"""

    frame_id: int
    road_mask: np.ndarray
    known_objects: List[DetectedObject]
    unknown_regions: List[UnknownRegion]
    risk_level: str
    major_reason: str
    anomaly_mask: Optional[np.ndarray] = None
    corridor_mask: Optional[np.ndarray] = None
    corridor_polygon: Optional[List[Tuple[int, int]]] = None
    corridor_centerline: Optional[List[Tuple[int, int]]] = None
    obstacle_risks: Optional[List[ObstacleRisk]] = None
    risk_system_status: str = SystemStatus.NORMAL
    risk_is_valid: bool = True
