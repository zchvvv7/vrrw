"""
文件名: result_visualizer.py
用途: 将道路区域、检测框和风险等级画到图像上
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import cv2
import numpy as np

from src.interface.schemas import FrameResult


# 绘制道路区域
def _draw_road_mask(output: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    road_overlay = np.zeros_like(output)
    road_overlay[:, :, 1] = road_mask
    return cv2.addWeighted(output, 0.7, road_overlay, 0.3, 0)


# 绘制已知障碍物检测框
def _draw_known_objects(output: np.ndarray, result: FrameResult) -> np.ndarray:
    for detected_object in result.known_objects:
        x1, y1, x2, y2 = detected_object.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = (
            f"{detected_object.class_name} "
            f"{detected_object.confidence:.2f}"
        )
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return output


# 绘制未知异常区域检测框
def _draw_unknown_regions(output: np.ndarray, result: FrameResult) -> np.ndarray:
    for unknown_region in result.unknown_regions:
        x1, y1, x2, y2 = unknown_region.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"unknown {unknown_region.score:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )
    return output


# 绘制风险等级
def _draw_risk_info(output: np.ndarray, result: FrameResult) -> np.ndarray:
    cv2.putText(
        output,
        f"Risk: {result.risk_level}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        output,
        f"Reason: {result.major_reason}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    return output


# 绘制单帧处理结果
def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    output = frame.copy()
    output = _draw_road_mask(output, result.road_mask)
    output = _draw_known_objects(output, result)
    output = _draw_unknown_regions(output, result)
    output = _draw_risk_info(output, result)
    return output