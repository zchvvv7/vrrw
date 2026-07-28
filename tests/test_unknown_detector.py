"""
文件名: test_unknown_detector.py
用途: 测试未知障碍物检测后处理和错误状态
作者: 张楚涵
创建日期: 2026-07-24
最后修改日期: 2026-07-28
"""

from typing import Tuple

import numpy as np

from src.interface.schemas import DetectedObject
from src.modules.unknown_detector import INFERENCE_ERROR
from src.modules.unknown_detector import INVALID_INPUT_ERROR
from src.modules.unknown_detector import SUCCESS
from src.modules.unknown_detector import UnknownDetector


class FakeMask2AnomalyBackend:
    """提供可控异常分数图的测试后端"""

    # 初始化测试异常分数图
    def __init__(self, score_map: np.ndarray) -> None:
        self._score_map = score_map

    # 获取测试模型版本
    @property
    def model_version(self) -> str:
        return "mask2anomaly-test"

    # 返回预设异常分数图
    def predict(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        return self._score_map.copy(), 1.0


class FailingMask2AnomalyBackend:
    """模拟推理异常的测试后端"""

    # 获取测试模型版本
    @property
    def model_version(self) -> str:
        return "mask2anomaly-failing-test"

    # 抛出模拟推理异常
    def predict(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        raise RuntimeError("simulated inference failure")


# 创建未知障碍物检测测试配置
def build_test_config() -> dict:
    return {
        "backend": "mask2anomaly",
        "post_processing": {
            "pixel_threshold": 0.5,
            "known_mask_threshold": 0.5,
            "min_area_ratio": 0.0001,
            "max_area_ratio": 0.5,
            "lower_roi_ratio": 0.0,
            "roi_dilate_kernel_size": 3,
            "morphology_kernel_size": 3,
            "region_score_quantile": 0.95,
        },
    }


# 测试异常分数图可以转换成未知障碍物区域
def test_predict_builds_unknown_region() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )
    road_mask = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )
    score_map = np.zeros(
        (100, 200),
        dtype=np.float32,
    )
    score_map[40:70, 80:110] = 0.9
    backend = FakeMask2AnomalyBackend(score_map)
    detector = UnknownDetector(
        build_test_config(),
        backend=backend,
    )

    result = detector.predict(
        frame,
        road_mask,
    )

    assert result.error_code == SUCCESS
    assert result.is_successful
    assert result.model_version == "mask2anomaly-test"
    assert result.anomaly_mask.dtype == np.uint8
    assert len(result.regions) == 1
    assert np.isclose(
        result.regions[0].score,
        0.9,
    )
    assert result.regions[0].area is not None
    assert result.regions[0].mask_rle is not None
    assert result.regions[0].mask_rle["size"] == [
        100,
        200,
    ]
    assert sum(
        result.regions[0].mask_rle["counts"]
    ) == 100 * 200


# 测试已知目标区域会从未知异常结果中排除
def test_predict_excludes_known_object_area() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )
    road_mask = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )
    score_map = np.zeros(
        (100, 200),
        dtype=np.float32,
    )
    score_map[40:70, 80:110] = 0.9
    known_object = DetectedObject(
        class_name="cone",
        bbox=(78, 38, 112, 72),
        confidence=0.9,
    )
    detector = UnknownDetector(
        build_test_config(),
        backend=FakeMask2AnomalyBackend(score_map),
    )

    result = detector.predict(
        frame,
        road_mask,
        known_objects=[known_object],
    )

    assert result.error_code == SUCCESS
    assert not np.any(result.anomaly_mask)
    assert result.regions == []


# 测试已知区域之外的异常仍会作为未知目标保留
def test_predict_keeps_anomaly_outside_known_objects() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )
    road_mask = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )
    score_map = np.zeros(
        (100, 200),
        dtype=np.float32,
    )
    score_map[20:40, 20:40] = 0.9
    score_map[60:80, 140:160] = 0.8
    known_object = DetectedObject(
        class_name="vehicle",
        bbox=(18, 18, 42, 42),
        confidence=0.95,
    )
    detector = UnknownDetector(
        build_test_config(),
        backend=FakeMask2AnomalyBackend(score_map),
    )

    result = detector.predict(
        frame,
        road_mask,
        known_objects=[known_object],
    )

    assert result.error_code == SUCCESS
    assert len(result.regions) == 1
    assert result.regions[0].bbox == (
        140,
        60,
        160,
        80,
    )
    assert np.isclose(
        result.regions[0].score,
        0.8,
    )


# 测试无效输入会返回明确错误码
def test_predict_rejects_invalid_road_mask() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )
    road_mask = np.zeros(
        (50, 100),
        dtype=np.uint8,
    )
    score_map = np.zeros(
        (100, 200),
        dtype=np.float32,
    )
    detector = UnknownDetector(
        build_test_config(),
        backend=FakeMask2AnomalyBackend(score_map),
    )

    result = detector.predict(
        frame,
        road_mask,
    )

    assert result.error_code == INVALID_INPUT_ERROR
    assert not result.is_successful


# 测试模型推理失败会返回明确错误码
def test_predict_reports_inference_failure() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )
    road_mask = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )
    detector = UnknownDetector(
        build_test_config(),
        backend=FailingMask2AnomalyBackend(),
    )

    result = detector.predict(
        frame,
        road_mask,
    )

    assert result.error_code == INFERENCE_ERROR
    assert not result.is_successful
    assert "simulated inference failure" in result.error_message
