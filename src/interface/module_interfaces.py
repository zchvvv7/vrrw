"""
文件名: module_interfaces.py
用途: 定义距离估计和视频行驶走廊预测模块的稳定接口
作者: 张楚涵
创建日期: 2026-07-28
最后修改日期: 2026-07-28
"""

from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np

from src.interface.schemas import CorridorPredictionResult
from src.interface.schemas import DetectedObject
from src.interface.schemas import DistanceEstimationResult


class DistanceEstimatorInterface(ABC):
    """规定已知障碍物距离估计模块必须实现的接口"""

    # 估计当前视频帧内所有已知障碍物的距离
    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        frame_id: int,
        known_objects: List[DetectedObject],
    ) -> DistanceEstimationResult:
        raise NotImplementedError


class CorridorPredictorInterface(ABC):
    """规定视频模式自车行驶走廊预测模块必须实现的接口"""

    # 根据当前视频帧和道路掩码预测图像空间行驶走廊
    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
    ) -> CorridorPredictionResult:
        raise NotImplementedError

    # 清空跨帧状态并准备处理新视频
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError