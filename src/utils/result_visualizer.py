"""
文件名: result_visualizer.py
用途: 将道路区域、检测框和风险等级画到图像上
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import cv2
import numpy as np

from src.interface.schemas import FrameResult


# TODO: 绘制单帧处理结果
def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    output = frame.copy()
    return output