"""
文件名: known_detector.py
用途: 检测已知类型道路障碍物
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

from typing import List

import numpy as np

from src.interface.schemas import DetectedObject


class KnownDetector:
    """负责检测锥桶、护栏、纸箱、轮胎等已知障碍物"""

    # TODO: 检测已知障碍物
    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        return []