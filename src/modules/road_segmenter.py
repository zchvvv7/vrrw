"""
文件名: road_segmenter.py
用途: 输出可行驶区域分割结果
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import cv2
import numpy as np


class RoadSegmenter:
    """负责识别图像中的可行驶区域"""

    # TODO: 预测可行驶区域
    def predict(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)