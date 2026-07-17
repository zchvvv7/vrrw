"""
文件名: schemas.py
用途: 定义系统各模块之间传递的数据结构
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-16
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
class UnknownRegion:
    """表示一个未知异常区域"""

    bbox: Tuple[int, int, int, int]
    score: float
    distance: Optional[float] = None


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

    @property
    def is_successful(self) -> bool:
        return self.error_code == 0

    @property
    def is_degraded(self) -> bool:
        return self.system_status == SystemStatus.DEGRADED

    @property
    def is_unavailable(self) -> bool:
        return self.system_status == SystemStatus.UNAVAILABLE

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
class FrameResult:
    """保存单帧处理结果"""

    frame_id: int
    road_mask: np.ndarray
    known_objects: List[DetectedObject]
    unknown_regions: List[UnknownRegion]
    risk_level: str
    major_reason: str