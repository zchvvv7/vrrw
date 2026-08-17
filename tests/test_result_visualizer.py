"""
文件名: test_result_visualizer.py
用途: 测试原始视频与右侧风险信息面板的可视化输出
作者: 张楚涵
创建日期: 2026-07-31
最后修改日期: 2026-08-17
"""

from typing import Any
from typing import List

import cv2
import numpy as np
import pytest

from src.interface.schemas import DetectedObject
from src.interface.schemas import FrameResult
from src.interface.schemas import UnknownRegion
from src.utils.live_visualizer import LiveVisualizer
from src.utils.result_visualizer import draw_result
from src.utils.result_visualizer import get_output_size


# 创建包含已知和未知障碍物的测试结果
def build_frame_result() -> FrameResult:
    return FrameResult(
        frame_id=0,
        road_mask=np.zeros(
            (720, 1280),
            dtype=np.uint8,
        ),
        known_objects=[
            DetectedObject(
                class_name="cone",
                bbox=(300, 300, 360, 500),
                confidence=0.90,
                distance=5.0,
            ),
            DetectedObject(
                class_name="vehicle",
                bbox=(500, 250, 800, 620),
                confidence=0.95,
                distance=3.3,
            ),
        ],
        unknown_regions=[
            UnknownRegion(
                bbox=(900, 350, 980, 560),
                score=0.82,
                distance=7.2,
            ),
        ],
        risk_level="danger",
        major_reason="known_obstacle_intersecting_critical_distance",
        risk_system_status="normal",
    )


# 测试输出尺寸包含与原视频等高的右侧面板
def test_output_size_includes_side_panel() -> None:
    assert get_output_size(1280, 720) == (1690, 720)


# 测试左侧原始画面不会被框、掩码或文字修改
def test_video_pixels_remain_unchanged() -> None:
    random_generator = np.random.default_rng(7)
    frame = random_generator.integers(
        0,
        256,
        size=(720, 1280, 3),
        dtype=np.uint8,
    )

    output = draw_result(
        frame,
        build_frame_result(),
    )

    assert output.shape == (720, 1690, 3)
    assert np.array_equal(
        output[:, : frame.shape[1]],
        frame,
    )
    assert np.any(output[:, frame.shape[1] :] != 0)


# 测试右侧面板显示障碍物距离和总风险
def test_panel_contains_distance_and_risk_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_text: List[str] = []
    original_put_text = cv2.putText

    # 记录OpenCV实际绘制到面板的文字
    def capture_text(
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        captured_text.append(str(args[1]))
        return original_put_text(*args, **kwargs)

    monkeypatch.setattr(
        cv2,
        "putText",
        capture_text,
    )
    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    draw_result(
        frame,
        build_frame_result(),
    )

    assert "Vehicle in 3.3m" in captured_text
    assert "Cone in 5m" in captured_text
    assert "Unknown obstacle in 7.2m" in captured_text
    assert "DANGER" in captured_text


# 测试实时窗口按完整界面的宽高比缩放
def test_live_visualizer_scales_complete_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = {}

    # 记录窗口创建参数
    def record_named_window(
        window_name: str,
        window_flag: int,
    ) -> None:
        events["window_name"] = window_name
        events["window_flag"] = window_flag

    # 记录窗口缩放尺寸
    def record_resize_window(
        window_name: str,
        width: int,
        height: int,
    ) -> None:
        events["resize_name"] = window_name
        events["resize_size"] = (width, height)

    # 记录传入实时窗口的完整帧尺寸
    def record_show_frame(
        window_name: str,
        frame: np.ndarray,
    ) -> None:
        events["show_name"] = window_name
        events["frame_shape"] = frame.shape

    # 模拟没有按下退出键
    def return_no_key(delay: int) -> int:
        events["delay"] = delay
        return -1

    # 记录指定窗口关闭事件
    def record_destroy_window(window_name: str) -> None:
        events["destroy_name"] = window_name

    monkeypatch.setattr(
        cv2,
        "namedWindow",
        record_named_window,
    )
    monkeypatch.setattr(
        cv2,
        "resizeWindow",
        record_resize_window,
    )
    monkeypatch.setattr(
        cv2,
        "imshow",
        record_show_frame,
    )
    monkeypatch.setattr(
        cv2,
        "waitKey",
        return_no_key,
    )
    monkeypatch.setattr(
        cv2,
        "destroyWindow",
        record_destroy_window,
    )
    visualizer = LiveVisualizer(
        max_window_width=500,
        max_window_height=400,
    )
    output_frame = np.zeros(
        (720, 1690, 3),
        dtype=np.uint8,
    )

    assert visualizer.show(output_frame)
    visualizer.close()

    assert events["frame_shape"] == output_frame.shape
    assert events["resize_size"] == (500, 213)
    assert events["destroy_name"] == "Road Risk Warning"
