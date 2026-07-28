"""
文件名: test_corridor_predictor.py
用途: 测试视频行驶走廊预测接口和占位实现
作者: 张楚涵
创建日期: 2026-07-28
最后修改日期: 2026-07-28
"""

from typing import List
from typing import Tuple

import numpy as np

from src.modules.corridor_predictor import SUCCESS
from src.modules.corridor_predictor import CorridorPredictor


class FakeCorridorPredictor(CorridorPredictor):
    """模拟能够生成图像空间走廊的实现"""

    # 返回与视频帧同尺寸的测试走廊
    def _predict_corridor(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        float,
    ]:
        corridor_mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8,
        )
        corridor_mask[20:, 30:50] = 255
        polygon = [
            (30, 39),
            (50, 39),
            (45, 20),
            (35, 20),
        ]
        centerline = [
            (40, 39),
            (40, 20),
        ]
        return (
            corridor_mask,
            polygon,
            centerline,
            0.8,
        )


# 创建测试视频帧和道路掩码
def build_inputs() -> Tuple[np.ndarray, np.ndarray]:
    frame = np.zeros(
        (40, 80, 3),
        dtype=np.uint8,
    )
    road_mask = np.full(
        (40, 80),
        255,
        dtype=np.uint8,
    )
    return frame, road_mask


# 测试关闭模块时不生成虚假的走廊数据
def test_disabled_predictor_returns_empty_result() -> None:
    frame, road_mask = build_inputs()
    predictor = CorridorPredictor(
        config={"enabled": False},
    )

    result = predictor.predict(
        frame=frame,
        frame_id=0,
        road_mask=road_mask,
    )

    assert result.error_code == SUCCESS
    assert result.is_successful
    assert not result.is_enabled
    assert result.corridor_mask is None
    assert result.polygon == []
    assert result.centerline == []


# 测试具体实现可以返回完整的图像空间走廊
def test_implementation_returns_corridor() -> None:
    frame, road_mask = build_inputs()
    predictor = FakeCorridorPredictor(
        config={
            "enabled": True,
            "method": "road_mask_temporal",
        },
    )

    result = predictor.predict(
        frame=frame,
        frame_id=2,
        road_mask=road_mask,
    )

    assert result.error_code == SUCCESS
    assert result.corridor_mask is not None
    assert result.corridor_mask.shape == frame.shape[:2]
    assert result.confidence == 0.8
    assert result.method == "road_mask_temporal"
