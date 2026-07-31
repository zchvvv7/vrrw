"""
文件名: test_result_visualizer.py
用途: 测试行驶走廊掩码、边界和中心线的可视化输出
作者: 张楚涵
创建日期: 2026-07-31
最后修改日期: 2026-07-31
"""

import numpy as np

from src.interface.schemas import FrameResult
from src.utils.result_visualizer import _draw_corridor_mask


# 创建包含完整走廊数据的测试帧结果
def build_frame_result() -> FrameResult:
    corridor_mask = np.zeros(
        (40, 80),
        dtype=np.uint8,
    )
    corridor_mask[10:31, 20:61] = 255
    return FrameResult(
        frame_id=0,
        road_mask=np.zeros(
            (40, 80),
            dtype=np.uint8,
        ),
        known_objects=[],
        unknown_regions=[],
        risk_level="safe",
        major_reason="no_obstacle",
        corridor_mask=corridor_mask,
        corridor_polygon=[
            (20, 10),
            (60, 10),
            (60, 30),
            (20, 30),
        ],
        corridor_centerline=[
            (40, 10),
            (40, 30),
        ],
    )


# 测试没有走廊掩码时保持图像不变
def test_missing_corridor_keeps_frame_unchanged() -> None:
    frame = np.zeros(
        (40, 80, 3),
        dtype=np.uint8,
    )
    result = build_frame_result()
    result.corridor_mask = None

    output = _draw_corridor_mask(frame.copy(), result)

    assert np.array_equal(output, frame)


# 测试走廊掩码、边界和中心线均会被绘制
def test_corridor_geometry_is_visible() -> None:
    frame = np.zeros(
        (40, 80, 3),
        dtype=np.uint8,
    )
    result = build_frame_result()

    output = _draw_corridor_mask(frame.copy(), result)

    assert output[20, 30, 0] > 0
    assert np.any(output[10, 20] > 0)
    assert np.all(output[20, 40] > 0)
