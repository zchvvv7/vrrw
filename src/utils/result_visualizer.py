"""
文件名: result_visualizer.py
用途: 将道路区域、检测框和风险等级画到图像上
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-08-03
"""

import cv2
import numpy as np

from src.interface.schemas import FrameResult
from src.interface.schemas import ObstacleRisk


RISK_COLORS = {
    "safe": (0, 200, 0),
    "notice": (0, 255, 255),
    "warning": (0, 140, 255),
    "danger": (0, 0, 255),
}


# 查找与当前检测框对应的单目标风险结果
def _find_obstacle_risk(
    result: FrameResult,
    source: str,
    bbox: tuple,
) -> ObstacleRisk | None:
    if result.obstacle_risks is None:
        return None
    for obstacle_risk in result.obstacle_risks:
        if (
            obstacle_risk.source == source
            and tuple(obstacle_risk.bbox) == tuple(bbox)
        ):
            return obstacle_risk
    return None


# 根据单目标风险等级选择绘制颜色
def _get_risk_color(
    obstacle_risk: ObstacleRisk | None,
    default_color: tuple,
) -> tuple:
    if obstacle_risk is None:
        return default_color
    return RISK_COLORS.get(
        obstacle_risk.risk_level,
        default_color,
    )


# 绘制道路区域
def _draw_road_mask(output: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    road_overlay = np.zeros_like(output)
    road_overlay[:, :, 1] = road_mask
    return cv2.addWeighted(output, 0.7, road_overlay, 0.3, 0)


# 绘制视频模式下预测的自车行驶走廊
def _draw_corridor_mask(
    output: np.ndarray,
    result: FrameResult,
) -> np.ndarray:
    if result.corridor_mask is None:
        return output
    if result.corridor_mask.shape != output.shape[:2]:
        return output

    corridor_pixels = result.corridor_mask > 0
    corridor_overlay = np.zeros_like(output)
    corridor_overlay[:, :, 0] = result.corridor_mask
    blended_output = cv2.addWeighted(
        output,
        0.75,
        corridor_overlay,
        0.25,
        0,
    )
    output[corridor_pixels] = blended_output[
        corridor_pixels
    ]

    if (
        result.corridor_polygon is not None
        and len(result.corridor_polygon) >= 3
    ):
        polygon = np.asarray(
            result.corridor_polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(
            output,
            [polygon],
            True,
            (255, 180, 0),
            2,
        )

    if (
        result.corridor_centerline is not None
        and len(result.corridor_centerline) >= 2
    ):
        centerline = np.asarray(
            result.corridor_centerline,
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(
            output,
            [centerline],
            False,
            (255, 255, 255),
            2,
        )
    return output


# 绘制未知异常像素区域
def _draw_anomaly_mask(
    output: np.ndarray,
    result: FrameResult,
) -> np.ndarray:
    if result.anomaly_mask is None:
        return output

    if result.anomaly_mask.shape != output.shape[:2]:
        return output

    anomaly_pixels = result.anomaly_mask > 0
    anomaly_overlay = np.zeros_like(output)
    anomaly_overlay[:, :, 2] = result.anomaly_mask
    blended_output = cv2.addWeighted(
        output,
        0.65,
        anomaly_overlay,
        0.35,
        0,
    )
    output[anomaly_pixels] = blended_output[
        anomaly_pixels
    ]
    return output


# 绘制已知障碍物检测框
def _draw_known_objects(output: np.ndarray, result: FrameResult) -> np.ndarray:
    for detected_object in result.known_objects:
        x1, y1, x2, y2 = detected_object.bbox
        obstacle_risk = _find_obstacle_risk(
            result,
            "known",
            detected_object.bbox,
        )
        color = _get_risk_color(
            obstacle_risk,
            (0, 255, 255),
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{detected_object.class_name} "
            f"{detected_object.confidence:.2f}"
        )
        if detected_object.distance is not None:
            label = (
                f"{label} "
                f"{detected_object.distance:.2f}m"
            )
        if obstacle_risk is not None:
            label = (
                f"{label} {obstacle_risk.spatial_relation} "
                f"{obstacle_risk.risk_level}"
            )
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return output


# 绘制未知异常区域检测框
def _draw_unknown_regions(
    output: np.ndarray,
    result: FrameResult,
) -> np.ndarray:
    for unknown_region in result.unknown_regions:
        x1, y1, x2, y2 = unknown_region.bbox
        obstacle_risk = _find_obstacle_risk(
            result,
            "unknown",
            unknown_region.bbox,
        )
        color = _get_risk_color(
            obstacle_risk,
            (255, 0, 255),
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"unknown {unknown_region.score:.2f}"
        if unknown_region.distance is not None:
            label = (
                f"{label} {unknown_region.distance:.2f}m"
            )
        if obstacle_risk is not None:
            label = (
                f"{label} {obstacle_risk.spatial_relation} "
                f"{obstacle_risk.risk_level}"
            )
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return output


# 绘制风险等级
def _draw_risk_info(output: np.ndarray, result: FrameResult) -> np.ndarray:
    risk_color = RISK_COLORS.get(
        result.risk_level,
        (0, 0, 255),
    )
    cv2.putText(
        output,
        f"Risk: {result.risk_level}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        risk_color,
        2,
    )
    cv2.putText(
        output,
        f"Reason: {result.major_reason}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        risk_color,
        2,
    )
    cv2.putText(
        output,
        f"System: {result.risk_system_status}",
        (30, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        risk_color,
        2,
    )
    if result.obstacle_risks is not None:
        valid_ttc_values = [
            item.ttc
            for item in result.obstacle_risks
            if item.ttc is not None
        ]
        if valid_ttc_values:
            cv2.putText(
                output,
                f"TTC: {min(valid_ttc_values):.2f}s",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                risk_color,
                2,
            )
    return output


# 绘制单帧处理结果
def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    output = frame.copy()
    output = _draw_road_mask(output, result.road_mask)
    output = _draw_corridor_mask(output, result)
    output = _draw_anomaly_mask(output, result)
    output = _draw_known_objects(output, result)
    output = _draw_unknown_regions(output, result)
    output = _draw_risk_info(output, result)
    return output
