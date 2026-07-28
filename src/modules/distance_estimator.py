"""
文件名: distance_estimator.py
用途: 提供仅针对已知障碍物的距离估计模块开发骨架
作者:
创建日期: 2026-07-28
最后修改日期: 2026-07-28
"""

from typing import List
from typing import Optional

import numpy as np

from src.interface.module_interfaces import DistanceEstimatorInterface
from src.interface.schemas import DetectedObject
from src.interface.schemas import DistanceEstimationResult


class DistanceEstimator(DistanceEstimatorInterface):
    """管理已知障碍物距离估计的配置、校验和统一返回值"""

    # 估计当前视频帧内所有已知障碍物的距离
    def estimate(
        self,
        frame: np.ndarray,
        frame_id: int,
        known_objects: List[DetectedObject],
    ) -> DistanceEstimationResult: