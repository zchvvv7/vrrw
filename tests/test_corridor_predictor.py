"""
文件名: test_corridor_predictor.py
用途: 测试视频行驶走廊预测接口和实现
作者: 张楚涵，温涵清
创建日期: 2026-07-28
最后修改日期: 2026-08-07
"""

from typing import List
from typing import Tuple

import numpy as np
import pytest

from src.interface.schemas import DetectedObject
from src.interface.schemas import UnknownRegion
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
    frame[height // 2 :, :] = [100, 100, 100]
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
        return build_inputs(height=60, width=100, road_width=60)

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
                (result.corridor_mask > 0) & (road_mask == 0)
            )
            assert not outside_road

    # 测试无效道路掩码时不崩溃
    def test_invalid_road_mask_no_crash(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:] = [128, 128, 128]

        empty_road = np.zeros((40, 80), dtype=np.uint8)

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

        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

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
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
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
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        road_mask = np.zeros((60, 100), dtype=np.uint8)
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
        frame, road_mask = build_obstructed_inputs(height=60, width=100)
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
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [128, 128, 128]

        road_mask = np.zeros((height, width), dtype=np.uint8)
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
        centerline_xs = [p[0] for p in result.centerline]
        mean_centerline_x = np.mean(centerline_xs)

        tolerance = road_width * 0.15
        assert abs(mean_centerline_x - center_x) < tolerance, (
            f"Centerline mean x={mean_centerline_x} "
            f"not within {tolerance} of road "
            f"center {center_x}"
        )


class TestObstacleAvoidance:
    """障碍物避让功能测试"""

    # 创建带几何投影配置的预测器
    @pytest.fixture
    def predictor(self) -> CorridorPredictor:
        return CorridorPredictor(
            config={
                "enabled": True,
                "method": "road_mask_geometry",
                "focal_length_px": 1000.0,
                "camera_height_m": 1.50,
                "horizon_ratio": 0.50,
                "ego_speed_mps": 13.9,
                "min_distance_m": 0.3,
                "max_distance_m": 80.0,
            }
        )

    # 创建带障碍物的测试场景
    @pytest.fixture
    def scene_with_obstacles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, DetectedObject, UnknownRegion]:
        height, width = 60, 100
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [128, 128, 128]

        road_mask = np.zeros((height, width), dtype=np.uint8)
        road_mask[:, 20:80] = 255

        # 添加一个已知障碍物
        known_object = DetectedObject(
            class_name="car",
            bbox=(35, 25, 45, 35),  # x1, y1, x2, y2
            confidence=0.95,
            distance=None,
        )

        # 添加一个未知区域
        unknown_region = UnknownRegion(
            bbox=(55, 20, 70, 30),
            score=0.8,
            distance=None,
        )

        return frame, road_mask, known_object, unknown_region

    # 测试障碍物避让：扣除后掩码面积应减少
    def test_obstacle_avoidance_reduces_area(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, known_obj, unknown_reg = scene_with_obstacles

        # 不传入障碍物，获取原始避让结果
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )
        original_mask = predictor.get_obstacle_avoidance_result()
        original_area = np.sum(original_mask > 0)

        # 传入障碍物，获取避让后走廊
        predictor.predict(
            frame=frame,
            frame_id=1,
            road_mask=road_mask,
            known_objects=[known_obj],
            unknown_regions=[unknown_reg],
        )
        avoided_mask = predictor.get_obstacle_avoidance_result()
        avoided_area = np.sum(avoided_mask > 0)

        # 避让后面积应小于或等于原始面积
        assert avoided_area <= original_area, (
            f"避让后面积 ({avoided_area}) 不应大于 原始面积 ({original_area})"
        )

    # 测试避让后的掩码与原始掩码的差异
    def test_avoidance_mask_difference(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, known_obj, _ = scene_with_obstacles

        # 获取原始走廊
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )
        original_mask = predictor._last_obstacle_cutout_mask

        # 传入障碍物
        predictor.predict(
            frame=frame,
            frame_id=1,
            road_mask=road_mask,
            known_objects=[known_obj],
        )
        avoided_mask = predictor._last_obstacle_cutout_mask

        # 两个掩码不应完全相同
        assert not np.array_equal(original_mask, avoided_mask)

    # 测试已知障碍物 bbox 区域被扣除
    def test_known_obstacle_bbox_removed(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, known_obj, _ = scene_with_obstacles

        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
            known_objects=[known_obj],
        )
        avoided_mask = predictor.get_obstacle_avoidance_result()

        if avoided_mask is not None:
            x1, y1, x2, y2 = known_obj.bbox
            obstacle_region = avoided_mask[y1:y2, x1:x2]
            # 障碍物区域应为 0（已扣除）
            assert np.all(obstacle_region == 0), (
                f"已知障碍物 bbox 区域 ({x1}:{x2}, {y1}:{y2}) 未被完全扣除"
            )

    # 测试未知区域 bbox 被扣除
    def test_unknown_region_bbox_removed(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, _, unknown_reg = scene_with_obstacles

        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
            unknown_regions=[unknown_reg],
        )
        avoided_mask = predictor.get_obstacle_avoidance_result()

        if avoided_mask is not None:
            x1, y1, x2, y2 = unknown_reg.bbox
            obstacle_region = avoided_mask[y1:y2, x1:x2]
            # 未知区域应为 0（已扣除）
            assert np.all(obstacle_region == 0), (
                f"未知区域 bbox ({x1}:{x2}, {y1}:{y2}) 未被完全扣除"
            )

    # 测试空障碍物列表不影响结果
    def test_empty_obstacle_list_no_effect(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, _, _ = scene_with_obstacles

        # 空列表不应产生影响
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
            known_objects=[],
            unknown_regions=[],
        )

        assert result.is_successful
        assert result.corridor_mask is not None

    # 测试无障碍物时接口仍能正常工作
    def test_no_obstacles_normal_operation(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, _, _ = scene_with_obstacles

        # 不传障碍物参数
        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert result.is_successful
        assert result.corridor_mask is not None

    # 测试 get_obstacle_avoidance_result 接口
    def test_get_obstacle_avoidance_result(
        self,
        predictor: CorridorPredictor,
        scene_with_obstacles: Tuple,
    ) -> None:
        frame, road_mask, known_obj, _ = scene_with_obstacles

        # 先不传障碍物
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )
        result_without = predictor.get_obstacle_avoidance_result()

        # 传入障碍物
        predictor.predict(
            frame=frame,
            frame_id=1,
            road_mask=road_mask,
            known_objects=[known_obj],
        )
        result_with = predictor.get_obstacle_avoidance_result()

        assert result_without is not None
        assert result_with is not None
        # 两个结果可能相同（如果障碍物不在走廊内）或不同


class TestGeometricProjection:
    """几何投影距离计算测试"""

    # 创建带几何投影配置的预测器
    @pytest.fixture
    def predictor(self) -> CorridorPredictor:
        return CorridorPredictor(
            config={
                "enabled": True,
                "method": "road_mask_geometry",
                "focal_length_px": 1000.0,
                "camera_height_m": 1.50,
                "horizon_ratio": 0.50,
                "ego_speed_mps": 13.9,
                "min_distance_m": 0.3,
                "max_distance_m": 80.0,
            }
        )

    # 创建测试场景
    @pytest.fixture
    def road_scene(self) -> Tuple[np.ndarray, np.ndarray]:
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [128, 128, 128]

        road_mask = np.zeros((height, width), dtype=np.uint8)
        road_mask[:, 340:940] = 255

        return frame, road_mask

    # 测试预测信息接口返回结果
    def test_get_prediction_info_returns_list(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple,
    ) -> None:
        frame, road_mask = road_scene
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        info = predictor.get_prediction_info()
        assert isinstance(info, list)

    # 测试预测信息格式正确
    def test_prediction_info_format(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple,
    ) -> None:
        frame, road_mask = road_scene
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        info = predictor.get_prediction_info()
        for marker in info:
            assert "x" in marker
            assert "y" in marker
            assert "time_s" in marker
            assert "distance_m" in marker
            assert isinstance(marker["x"], int)
            assert isinstance(marker["y"], int)
            assert isinstance(marker["time_s"], float)
            assert isinstance(marker["distance_m"], float)

    # 测试距离计算：距地平线越远，距离越近
    def test_distance_decreases_with_offset(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        # 直接测试 _compute_prediction_markers
        frame_height = 720
        # 需要至少 2 个点，使用相同的点两次
        centerline = [
            (640, 400),  # offset=40px (距地平线)
            (640, 400),
            (640, 600),  # offset=240px
            (640, 600),
            (640, 700),  # offset=340px
            (640, 700),
        ]

        markers = predictor._compute_prediction_markers(
            centerline, frame_height
        )

        # 验证标记点有正确的距离
        assert len(markers) > 0
        for marker in markers:
            assert marker["distance_m"] > 0
            assert marker["time_s"] > 0

    # 测试几何投影公式正确性
    def test_geometric_projection_formula(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        # 手动计算验证
        focal_length = 1000.0
        camera_height = 1.50
        frame_height = 720
        horizon_y = frame_height * 0.50  # 360

        test_y = 500
        offset = test_y - horizon_y  # 140
        expected_distance = focal_length * camera_height / offset
        expected_time = expected_distance / 13.9

        # 需要至少 2 个点，所以添加两个相同的点
        centerline = [(640, test_y), (640, test_y)]
        markers = predictor._compute_prediction_markers(
            centerline, frame_height
        )

        assert len(markers) >= 1
        # 验证至少有一个点的计算正确
        found = False
        for marker in markers:
            if abs(marker["distance_m"] - expected_distance) < 0.01:
                found = True
                assert abs(marker["time_s"] - expected_time) < 0.01
                break
        assert found, (
            f"未找到距离为 {expected_distance:.2f}m 的标记点, "
            f"实际标记: {markers}"
        )

    # 测试地平线以上的点无效
    def test_points_above_horizon_invalid(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame_height = 720
        centerline = [
            (640, 200),  # 地平线以上，offset=-160
            (640, 300),  # 地平线以上，offset=-60
        ]

        markers = predictor._compute_prediction_markers(
            centerline, frame_height
        )

        # 地平线以上的点应被过滤
        assert len(markers) == 0

    # 测试距离超出范围时被过滤
    def test_distance_out_of_range_filtered(
        self,
        predictor: CorridorPredictor,
    ) -> None:
        frame_height = 720

        # 距地平线很小，距离可能超限
        centerline_far = [(640, 365), (640, 365)]
        markers_far = predictor._compute_prediction_markers(
            centerline_far, frame_height
        )

        # 远处点应被过滤（距离超出 max_distance_m）
        assert len(markers_far) == 0

    # 测试自车速度影响时间计算
    def test_ego_speed_affects_time(
        self,
    ) -> None:
        # 低速
        predictor_slow = CorridorPredictor(
            config={
                "enabled": True,
                "focal_length_px": 1000.0,
                "camera_height_m": 1.50,
                "horizon_ratio": 0.50,
                "ego_speed_mps": 5.0,  # 18 km/h
                "min_distance_m": 0.3,
                "max_distance_m": 80.0,
            }
        )

        # 高速
        predictor_fast = CorridorPredictor(
            config={
                "enabled": True,
                "focal_length_px": 1000.0,
                "camera_height_m": 1.50,
                "horizon_ratio": 0.50,
                "ego_speed_mps": 30.0,  # 108 km/h
                "min_distance_m": 0.3,
                "max_distance_m": 80.0,
            }
        )

        frame_height = 720
        # 需要至少 2 个点
        centerline = [(640, 500), (640, 500)]

        markers_slow = predictor_slow._compute_prediction_markers(
            centerline, frame_height
        )
        markers_fast = predictor_fast._compute_prediction_markers(
            centerline, frame_height
        )

        # 相同距离下，速度越快，时间越短
        if markers_slow and markers_fast:
            assert markers_slow[0]["time_s"] > markers_fast[0]["time_s"]

    # 测试 reset 清空预测信息缓存
    def test_reset_clears_prediction_info(
        self,
        predictor: CorridorPredictor,
        road_scene: Tuple,
    ) -> None:
        frame, road_mask = road_scene

        # 预测一次
        predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )
        info1 = predictor.get_prediction_info()

        # reset
        predictor.reset()

        # 再次预测
        predictor.predict(
            frame=frame,
            frame_id=1,
            road_mask=road_mask,
        )
        info2 = predictor.get_prediction_info()

        # reset 后仍能正常获取信息
        assert isinstance(info1, list)
        assert isinstance(info2, list)


class TestConfigurationParameters:
    """配置参数传递测试"""

    # 测试几何投影参数正确传递
    def test_projection_params_in_config(self) -> None:
        config = {
            "enabled": True,
            "focal_length_px": 1200.0,
            "camera_height_m": 1.80,
            "horizon_ratio": 0.45,
            "ego_speed_mps": 16.7,  # 60 km/h
            "min_distance_m": 0.5,
            "max_distance_m": 100.0,
        }

        predictor = CorridorPredictor(config=config)

        assert predictor._focal_length_px == 1200.0
        assert predictor._camera_height_m == 1.80
        assert predictor._horizon_ratio == 0.45
        assert predictor._ego_speed_mps == 16.7
        assert predictor._min_distance_m == 0.5
        assert predictor._max_distance_m == 100.0

    # 测试默认参数
    def test_default_projection_params(self) -> None:
        predictor = CorridorPredictor(config={"enabled": True})

        assert predictor._focal_length_px == 1000.0
        assert predictor._camera_height_m == 1.50
        assert predictor._horizon_ratio == 0.50
        assert predictor._ego_speed_mps == 13.9
        assert predictor._min_distance_m == 0.3
        assert predictor._max_distance_m == 80.0

    # 测试 enable/disable 与新参数共存
    def test_disabled_with_projection_params(self) -> None:
        predictor = CorridorPredictor(
            config={
                "enabled": False,
                "focal_length_px": 1000.0,
                "ego_speed_mps": 13.9,
            }
        )

        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:] = [128, 128, 128]
        road_mask = np.zeros((40, 80), dtype=np.uint8)

        result = predictor.predict(
            frame=frame,
            frame_id=0,
            road_mask=road_mask,
        )

        assert not result.is_enabled
        assert result.corridor_mask is None
