"""
文件名: test_distance_estimator.py
用途: 测试已知障碍物距离估计接口和占位实现
作者: 张楚涵
创建日期: 2026-07-28
最后修改日期: 2026-07-28
"""

from dataclasses import replace
from inspect import signature
from typing import List

import numpy as np

from src.interface.schemas import DetectedObject
from src.modules.distance_estimator import INFERENCE_ERROR
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


# 测试未实现算法在启用后返回明确错误
def test_unimplemented_estimator_returns_error() -> None:
    estimator = DistanceEstimator(
        config={"enabled": True},
    )

    result = estimator.estimate(
        frame=build_frame(),
        frame_id=0,
        known_objects=build_known_objects(),
    )

    assert result.error_code == INFERENCE_ERROR
    assert "NotImplementedError" in result.error_message
    assert result.known_objects[0].distance is None
