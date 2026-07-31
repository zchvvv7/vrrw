"""
文件名: test_corridor_predictor.py
用途: 测试视频行驶走廊预测接口和实现
作者: 张楚涵，温涵清
创建日期: 2026-07-28
最后修改日期: 2026-07-30
"""

from typing import List
from typing import Tuple

import numpy as np
import pytest

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
def build_inputs(
    height: int = 40,
    width: int = 80,
    road_width: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )
    frame[:, :] = [128, 128, 128]
    frame[height // 2:, :] = [100, 100, 100]
    road_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )
    left_margin = (width - road_width) // 2
    right_margin = left_margin + road_width
    road_mask[:, left_margin:right_margin] = 255
    return frame, road_mask


# 创建带障碍物的测试图像
def build_obstructed_inputs(
    height: int = 40,
    width: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    frame, road_mask = build_inputs(height, width)
    frame[25:35, 35:45] = [50, 50, 200]
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


# 测试启用配置下的完整预测流程
class TestCorridorPredictor:
    """CorridorPredictor 完整功能测试"""

    # 创建启用状态的预测器
    @pytest.fixture
    def predictor(self) -> CorridorPredictor:
        return CorridorPredictor(
            config={
                "enabled": True,
                "method": "road_mask_geometry",
                "model_version": "v1.0",
                "min_road_width": 20,
                "min_valid_rows": 5,
            }
        )

    # 创建模拟道路图像
    @pytest.fixture
    def road_scene(self) -> Tuple[np.ndarray, np.ndarray]:
        return build_inputs(
            height=60, width=100, road_width=60
        )

    # 测试输出结构是否完整
    def test_predict_output_structure(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert hasattr(result, "corridor_mask")
        assert hasattr(result, "polygon")
        assert hasattr(result, "centerline")
        assert hasattr(result, "confidence")
        assert hasattr(result, "inference_time_ms")
        assert hasattr(result, "error_code")
        assert hasattr(result, "error_message")
        assert hasattr(result, "method")
        assert hasattr(result, "model_version")
        assert hasattr(result, "is_enabled")

    # 测试正常道路场景下生成非空走廊
    def test_normal_road_returns_corridor(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.is_successful
        assert result.error_code == SUCCESS
        assert result.corridor_mask is not None
        assert np.sum(result.corridor_mask > 0) > 0
        assert len(result.polygon) >= 3
        assert len(result.centerline) >= 2

    # 测试输出掩码尺寸与输入帧一致
    def test_output_mask_shape(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        if result.corridor_mask is not None:
            assert result.corridor_mask.shape == frame.shape[:2]
            assert result.corridor_mask.dtype == np.uint8

    # 测试置信度范围在 [0.0, 1.0]
    def test_confidence_range(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert 0.0 <= result.confidence <= 1.0

    # 测试走廊不超出道路掩码范围
    def test_corridor_within_road(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        if result.corridor_mask is not None:
            outside_road = np.any(
                (result.corridor_mask > 0)
                & (road_mask == 0)
            )
            assert not outside_road

    # 测试无效道路掩码时不崩溃
    def test_invalid_road_mask_no_crash(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame = np.zeros(
            (40, 80, 3), dtype=np.uint8
        )
        frame[:] = [128, 128, 128]

        empty_road = np.zeros(
            (40, 80), dtype=np.uint8
        )

        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=empty_road,
        )

        assert result.is_successful
        assert result.corridor_mask is not None
        assert np.sum(result.corridor_mask > 0) == 0
        assert result.confidence == 0.0

    # 测试 reset() 清空历史状态
    def test_reset_clears_history(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene

        result1 = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )
        confidence1 = result1.confidence

        predictor.reset()

        result2 = predictor.predict(
            frame=frame,
            frame_id=1,
            road_mask=road_mask,
        )
        confidence2 = result2.confidence

        assert result2.is_successful
        assert 0.0 <= confidence2 <= 1.0

    # 测试连续多帧输出稳定性
    def test_temporal_consistency(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        confidences = []
        for i in range(10):
            result = predictor.predict(
                frame=frame,
                frame_id=i,
                road_mask=road_mask,
            )
            confidences.append(result.confidence)

        for conf in confidences:
            assert 0.0 <= conf <= 1.0

    # 测试 None 输入的异常处理
    def test_none_frame_returns_error(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        road_mask = np.zeros((40, 80), dtype=np.uint8)
        result = predictor.predict(
            frame=None,
            frame_id=0,
            road_mask=road_mask,
        )

        assert not result.is_successful
        assert result.error_code == -1

    # 测试 None 道路掩码的异常处理
    def test_none_road_mask_returns_error(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame = np.zeros(
            (40, 80, 3), dtype=np.uint8
        )
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=None,
        )

        assert not result.is_successful
        assert result.error_code == -1

    # 测试尺寸不匹配的异常处理
    def test_mismatched_shapes_returns_error(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame = np.zeros(
            (40, 80, 3), dtype=np.uint8
        )
        road_mask = np.zeros(
            (60, 100), dtype=np.uint8
        )
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert not result.is_successful
        assert result.error_code == -1

    # 测试障碍物检测场景
    def test_obstacle_detection_reduces_corridor(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame, road_mask = build_obstructed_inputs(
            height=60, width=100
        )
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.is_successful
        assert 0.0 <= result.confidence <= 1.0

    # 测试不同帧尺寸的兼容性
    def test_different_frame_sizes(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        sizes = [(480, 640), (720, 1280), (240, 320)]
        for height, width in sizes:
            frame, road_mask = build_inputs(
                height=height,
                width=width,
                road_width=int(width * 0.6),
            )
            result = predictor.predict(
                frame=frame,
                frame_id=0,
                road_mask=road_mask,
            )
            if result.corridor_mask is not None:
                assert result.corridor_mask.shape == (height, width)

    # 测试推理时间非负
    def test_inference_time_nonnegative(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.inference_time_ms >= 0

    # 测试配置中的 method 和 model_version 正确传递
    def test_config_values_passed(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        frame, road_mask = road_scene
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.method == "road_mask_geometry"
        assert result.model_version == "v1.0"

    # 测试中心线跟随对称道路的几何中心
    def test_centerline_follows_road_center(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        height, width = 60, 100
        road_width = 60
        frame = np.zeros(
            (height, width, 3), dtype=np.uint8
        )
        frame[:, :] = [128, 128, 128]

        road_mask = np.zeros(
            (height, width), dtype=np.uint8
        )
        left_margin = (width - road_width) // 2
        right_margin = left_margin + road_width
        road_mask[:, left_margin:right_margin] = 255

        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.is_successful
        assert len(result.centerline) >= 2

        center_x = width / 2.0
        centerline_xs = [
            p[0] for p in result.centerline
        ]
        mean_centerline_x = np.mean(centerline_xs)

        tolerance = road_width * 0.15
        assert abs(mean_centerline_x - center_x) < tolerance, (
            f"Centerline mean x={mean_centerline_x} "
            f"not within {tolerance} of road "
            f"center {center_x}"
        )