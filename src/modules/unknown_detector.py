"""
文件名: unknown_detector.py
用途: 检测未知异常区域
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

from typing import List

import numpy as np

from src.interface.schemas import UnknownRegion


class UnknownDetector:
    """负责发现训练类别之外的异常道路风险"""

    # TODO: 检测未知异常区域
    def predict(self, frame: np.ndarray) -> List[UnknownRegion]:
        return []