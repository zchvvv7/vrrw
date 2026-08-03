"""
文件名: test_distance_estimator.py
用途: 测试已知障碍物距离估计
作者: 张楚涵，周子懿
创建日期: 2026-07-28
最后修改日期: 2026-08-01
"""

from dataclasses import replace
from inspect import signature
from typing import List

import numpy as np

from src.interface.schemas import DetectedObject
from src.modules.distance_estimator import SUCCESS
from src.modules.distance_estimator import DistanceEstimator


class FakeDistanceEstimator(DistanceEstimator):
    """模拟能够为已知障碍物写入距离的实现"""

    # 为每个测试已知障碍物写入固定距离
    def _estimate_known_objects(
        self,
        frame: np.ndarray,
        frame_id: int,
        known_objects: List[DetectedObject],
    ) -> List[DetectedObject]:
        return [
            replace(item, distance=12.5)
            for item in known_objects
        ]


# 创建测试视频帧
def build_frame() -> np.ndarray:
    return np.zeros(
        (40, 80, 3),
        dtype=np.uint8,
    )


# 创建测试已知障碍物
def build_known_objects() -> List[DetectedObject]:
    return [
        DetectedObject(
            class_name="cone",
            bbox=(10, 10, 20, 30),
            confidence=0.9,
        )
    ]


# 测试距离接口没有未知障碍物参数
def test_interface_only_accepts_known_objects() -> None:
    parameters = signature(
        DistanceEstimator.estimate
    ).parameters

    assert "known_objects" in parameters
    assert "unknown_regions" not in parameters


# 测试关闭模块时保持已知检测结果不变
def test_disabled_estimator_preserves_known_objects() -> None:
    known_objects = build_known_objects()
    estimator = DistanceEstimator(
        config={"enabled": False},
    )

    result = estimator.estimate(
        frame=build_frame(),
        frame_id=0,
        known_objects=known_objects,
    )

    assert result.error_code == SUCCESS
    assert result.is_successful
    assert not result.is_enabled
    assert result.known_objects == known_objects
    assert result.known_objects[0].distance is None


# 测试具体实现可按顺序写回已知障碍物距离
def test_implementation_writes_known_object_distance() -> None:
    estimator = FakeDistanceEstimator(
        config={
            "enabled": True,
            "method": "fake_metric_depth",
        },
    )

    result = estimator.estimate(
        frame=build_frame(),
        frame_id=3,
        known_objects=build_known_objects(),
    )

    assert result.error_code == SUCCESS
    assert result.known_objects[0].distance == 12.5
    assert result.method == "fake_metric_depth"


# 构造启用几何投影的估计器
def build_estimator() -> DistanceEstimator:
    return DistanceEstimator(
        config={
            "enabled": True,
            "method": "geometric_bbox_height",
            "model_version": "v1.0",
            "focal_length_px": 1000.0,
            "min_bbox_height_px": 8,
            "min_distance_m": 0.3,
            "max_distance_m": 80.0,
            "class_heights_m": {
                "cone": 0.70,
                "barrier": 1.00,
                "vehicle": 1.50,
            },
        },
    )


# 测试空目标列表可正常运行
def test_empty_known_objects() -> None:
    result = build_estimator().estimate(
        frame=build_frame(),
        frame_id=0,
        known_objects=[],
    )
    assert result.error_code == SUCCESS
    assert result.known_objects == []


# 测试多目标数量顺序不变且距离可写
def test_multiple_objects_preserve_order() -> None:
    objects = [
        DetectedObject(
            class_name="cone",
            bbox=(10, 10, 20, 30),
            confidence=0.9,
        ),
        DetectedObject(
            class_name="vehicle",
            bbox=(30, 5, 50, 35),
            confidence=0.8,
        ),
    ]
    result = build_estimator().estimate(
        frame=build_frame(),
        frame_id=1,
        known_objects=objects,
    )
    assert result.error_code == SUCCESS
    assert len(result.known_objects) == 2
    assert (
        result.known_objects[0].class_name == "cone"
    )
    assert (
        result.known_objects[1].class_name
        == "vehicle"
    )
    assert (
        result.known_objects[0].bbox
        == objects[0].bbox
    )
    assert (
        result.known_objects[0].distance is not None
    )
    assert (
        result.known_objects[1].distance is not None
    )


# 测试边界框过小或越界时不崩溃
def test_invalid_bbox_returns_none_distance() -> None:
    objects = [
        DetectedObject(
            class_name="cone",
            bbox=(10, 10, 20, 12),
            confidence=0.9,
        ),
        DetectedObject(
            class_name="cone",
            bbox=(-5, -5, 5, 5),
            confidence=0.8,
        ),
    ]
    result = build_estimator().estimate(
        frame=build_frame(),
        frame_id=2,
        known_objects=objects,
    )
    assert result.error_code == SUCCESS
    assert result.known_objects[0].distance is None
    assert result.known_objects[1].distance is None


# 测试无物体高度类别可以回退到路面接地点距离
def test_missing_class_height_uses_ground_projection() -> None:
    frame = np.zeros(
        (200, 100, 3),
        dtype=np.uint8,
    )
    pothole = DetectedObject(
        class_name="pothole",
        bbox=(30, 120, 70, 160),
        confidence=0.9,
    )

    result = build_estimator().estimate(
        frame=frame,
        frame_id=3,
        known_objects=[pothole],
    )

    assert result.error_code == SUCCESS
    assert result.known_objects[0].distance is not None


# 测试近中远距离绝对误差和相对误差
def test_near_mid_far_distance_errors() -> None:
    # 选用整数框高，使真值距离恰好为 f*H/h
    cases = [
        ("near", 140, 5.0),
        ("mid", 35, 20.0),
        ("far", 14, 50.0),
    ]
    objects = []
    for _name, bbox_height, _true_distance in cases:
        objects.append(
            DetectedObject(
                class_name="cone",
                bbox=(
                    10,
                    10,
                    20,
                    10 + bbox_height,
                ),
                confidence=0.9,
            )
        )

    frame = np.zeros(
        (200, 80, 3),
        dtype=np.uint8,
    )
    result = build_estimator().estimate(
        frame=frame,
        frame_id=4,
        known_objects=objects,
    )
    assert result.error_code == SUCCESS

    for index, (_name, _h, true_distance) in enumerate(cases):
        predicted = result.known_objects[index].distance
        assert predicted is not None
        abs_error = abs(predicted - true_distance)
        rel_error = abs_error / true_distance
        assert abs_error < 0.5
        assert rel_error < 0.05
