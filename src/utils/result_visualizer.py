"""
文件名: result_visualizer.py
用途: 在原始视频右侧生成障碍物与风险信息面板
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-08-17
"""

from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from src.interface.schemas import FrameResult
from src.interface.schemas import ObstacleRisk


PANEL_WIDTH_RATIO = 0.32
MIN_PANEL_WIDTH = 320
MAX_PANEL_WIDTH = 480
PANEL_BACKGROUND = (24, 27, 34)
PANEL_CARD_BACKGROUND = (35, 39, 48)
PANEL_BORDER = (66, 72, 84)
PRIMARY_TEXT = (240, 242, 245)
SECONDARY_TEXT = (163, 171, 184)
RISK_COLORS = {
    "safe": (88, 201, 116),
    "notice": (79, 194, 247),
    "warning": (59, 157, 255),
    "danger": (68, 68, 239),
}


# 判断距离值是否可以显示
def _is_valid_distance(distance: Optional[float]) -> bool:
    if distance is None:
        return False
    return bool(np.isfinite(distance) and distance > 0.0)


# 将距离格式化为紧凑的米制文本
def _format_distance(distance: Optional[float]) -> str:
    if not _is_valid_distance(distance):
        return "distance unavailable"
    distance_text = f"{float(distance):.1f}".rstrip("0").rstrip(".")
    return f"{distance_text}m"


# 将类别名称转换为面板显示名称
def _format_object_name(class_name: str) -> str:
    normalized_name = class_name.replace("_", " ").strip()
    if not normalized_name:
        return "Object"
    return normalized_name.title()


# 查找检测结果对应的单目标风险
def _find_obstacle_risk(
    result: FrameResult,
    source: str,
    bbox: tuple,
) -> Optional[ObstacleRisk]:
    if result.obstacle_risks is None:
        return None
    for obstacle_risk in result.obstacle_risks:
        if obstacle_risk.source == source and tuple(
            obstacle_risk.bbox
        ) == tuple(bbox):
            return obstacle_risk
    return None


# 获取风险等级对应的面板颜色
def _get_risk_color(risk_level: str) -> Tuple[int, int, int]:
    return RISK_COLORS.get(
        risk_level,
        SECONDARY_TEXT,
    )


# 汇总需要显示在面板中的障碍物
def _build_obstacle_items(
    result: FrameResult,
) -> List[Tuple[str, Optional[float], str]]:
    obstacle_items = []
    for detected_object in result.known_objects:
        obstacle_risk = _find_obstacle_risk(
            result,
            "known",
            detected_object.bbox,
        )
        risk_level = (
            obstacle_risk.risk_level if obstacle_risk is not None else "notice"
        )
        obstacle_items.append(
            (
                _format_object_name(detected_object.class_name),
                detected_object.distance,
                risk_level,
            )
        )

    for unknown_region in result.unknown_regions:
        obstacle_risk = _find_obstacle_risk(
            result,
            "unknown",
            unknown_region.bbox,
        )
        risk_level = (
            obstacle_risk.risk_level if obstacle_risk is not None else "notice"
        )
        obstacle_items.append(
            (
                "Unknown obstacle",
                unknown_region.distance,
                risk_level,
            )
        )

    obstacle_items.sort(
        key=lambda item: (
            not _is_valid_distance(item[1]),
            (float(item[1]) if _is_valid_distance(item[1]) else float("inf")),
            item[0],
        )
    )
    return obstacle_items


# 根据视频宽度计算右侧面板宽度
def _get_panel_width(frame_width: int) -> int:
    requested_width = int(round(frame_width * PANEL_WIDTH_RATIO))
    panel_width = max(
        MIN_PANEL_WIDTH,
        min(MAX_PANEL_WIDTH, requested_width),
    )
    if (frame_width + panel_width) % 2 != 0:
        panel_width += 1
    return panel_width


# 返回拼接信息面板后的输出视频尺寸
def get_output_size(
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame width and height must be positive.")
    return (
        frame_width + _get_panel_width(frame_width),
        frame_height,
    )


# 将过长文本截断到指定像素宽度
def _fit_text(
    text: str,
    max_width: int,
    font_scale: float,
    thickness: int,
) -> str:
    text_width = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][0]
    if text_width <= max_width:
        return text

    suffix = "..."
    shortened_text = text
    while shortened_text:
        shortened_text = shortened_text[:-1]
        candidate = shortened_text.rstrip() + suffix
        candidate_width = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        if candidate_width <= max_width:
            return candidate
    return suffix


# 绘制面板中的一行文字
def _draw_text(
    panel: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
    max_width: int,
) -> None:
    fitted_text = _fit_text(
        text,
        max_width,
        font_scale,
        thickness,
    )
    cv2.putText(
        panel,
        fitted_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# 绘制总风险卡片并返回后续内容起始纵坐标
def _draw_risk_card(
    panel: np.ndarray,
    result: FrameResult,
    scale: float,
    padding: int,
) -> int:
    panel_width = panel.shape[1]
    card_top = int(round(66 * scale))
    card_height = int(round(112 * scale))
    card_bottom = min(
        panel.shape[0] - 1,
        card_top + card_height,
    )
    cv2.rectangle(
        panel,
        (padding, card_top),
        (panel_width - padding, card_bottom),
        PANEL_CARD_BACKGROUND,
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        panel,
        (padding, card_top),
        (panel_width - padding, card_bottom),
        PANEL_BORDER,
        thickness=1,
    )

    label_y = card_top + int(round(30 * scale))
    risk_y = card_top + int(round(79 * scale))
    text_width = panel_width - 2 * padding - 24
    _draw_text(
        panel,
        "OVERALL RISK",
        (padding + 12, label_y),
        SECONDARY_TEXT,
        max(0.42, 0.52 * scale),
        1,
        text_width,
    )
    _draw_text(
        panel,
        result.risk_level.upper(),
        (padding + 12, risk_y),
        _get_risk_color(result.risk_level),
        max(0.68, 1.02 * scale),
        2,
        text_width,
    )
    return card_bottom + int(round(42 * scale))


# 绘制障碍物距离列表
def _draw_obstacle_list(
    panel: np.ndarray,
    result: FrameResult,
    start_y: int,
    scale: float,
    padding: int,
) -> None:
    panel_width = panel.shape[1]
    panel_height = panel.shape[0]
    content_width = panel_width - 2 * padding
    _draw_text(
        panel,
        "DETECTED OBJECTS",
        (padding, start_y),
        SECONDARY_TEXT,
        max(0.42, 0.52 * scale),
        1,
        content_width,
    )

    row_height = max(34, int(round(52 * scale)))
    row_y = start_y + max(30, int(round(39 * scale)))
    footer_space = max(38, int(round(52 * scale)))
    available_height = max(
        0,
        panel_height - footer_space - row_y,
    )
    max_rows = max(1, available_height // row_height)
    obstacle_items = _build_obstacle_items(result)

    if not obstacle_items:
        _draw_text(
            panel,
            "No obstacles detected",
            (padding, row_y),
            PRIMARY_TEXT,
            max(0.48, 0.62 * scale),
            1,
            content_width,
        )
        return

    visible_items = obstacle_items[:max_rows]
    hidden_count = len(obstacle_items) - len(visible_items)
    if hidden_count > 0 and len(visible_items) > 1:
        visible_items = visible_items[:-1]
        hidden_count = len(obstacle_items) - len(visible_items)

    for object_name, distance, risk_level in visible_items:
        distance_text = _format_distance(distance)
        if _is_valid_distance(distance):
            item_text = f"{object_name} in {distance_text}"
        else:
            item_text = f"{object_name}: {distance_text}"
        marker_radius = max(4, int(round(5 * scale)))
        cv2.circle(
            panel,
            (padding + marker_radius, row_y - marker_radius),
            marker_radius,
            _get_risk_color(risk_level),
            thickness=cv2.FILLED,
        )
        text_x = padding + marker_radius * 3
        _draw_text(
            panel,
            item_text,
            (text_x, row_y),
            PRIMARY_TEXT,
            max(0.48, 0.66 * scale),
            1,
            panel_width - padding - text_x,
        )
        row_y += row_height

    if hidden_count > 0:
        _draw_text(
            panel,
            f"+{hidden_count} more",
            (padding, row_y),
            SECONDARY_TEXT,
            max(0.44, 0.56 * scale),
            1,
            content_width,
        )


# 创建与视频画面同高的右侧信息面板
def _create_information_panel(
    frame_height: int,
    frame_width: int,
    result: FrameResult,
    dtype: np.dtype,
) -> np.ndarray:
    panel_width = _get_panel_width(frame_width)
    panel = np.full(
        (frame_height, panel_width, 3),
        PANEL_BACKGROUND,
        dtype=dtype,
    )
    panel[:, :2] = PANEL_BORDER
    scale = max(0.65, min(1.25, frame_height / 720.0))
    padding = max(20, int(round(panel_width * 0.075)))
    _draw_text(
        panel,
        "ROAD RISK MONITOR",
        (padding, max(30, int(round(38 * scale)))),
        PRIMARY_TEXT,
        max(0.52, 0.72 * scale),
        2,
        panel_width - 2 * padding,
    )
    list_start_y = _draw_risk_card(
        panel,
        result,
        scale,
        padding,
    )
    _draw_obstacle_list(
        panel,
        result,
        list_start_y,
        scale,
        padding,
    )
    _draw_text(
        panel,
        f"System: {result.risk_system_status.upper()}",
        (
            padding,
            frame_height
            - max(
                12,
                int(round(18 * scale)),
            ),
        ),
        SECONDARY_TEXT,
        max(0.38, 0.48 * scale),
        1,
        panel_width - 2 * padding,
    )
    return panel


# 将原始视频与右侧风险信息面板拼接为输出帧
def draw_result(
    frame: np.ndarray,
    result: FrameResult,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have shape H x W x 3.")
    panel = _create_information_panel(
        frame_height=frame.shape[0],
        frame_width=frame.shape[1],
        result=result,
        dtype=frame.dtype,
    )
    return np.concatenate(
        (frame.copy(), panel),
        axis=1,
    )
