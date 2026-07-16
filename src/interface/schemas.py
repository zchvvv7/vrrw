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


@dataclass
class FrameResult:
    """保存单帧处理结果"""

    frame_id: int
    road_mask: np.ndarray
    known_objects: List[DetectedObject]
    unknown_regions: List[UnknownRegion]
    risk_level: str
    major_reason: str