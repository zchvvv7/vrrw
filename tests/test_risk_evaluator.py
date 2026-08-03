"""
文件名: test_risk_evaluator.py
用途: 测试仅基于视频的空间冲突、TTC和风险防抖
作者: 张楚涵
创建日期: 2026-08-03
最后修改日期: 2026-08-03
"""

from typing import List

import numpy as np
import pytest

from src.interface.schemas import CorridorPredictionResult
from src.interface.schemas import DetectedObject
from src.interface.schemas import SystemStatus
from src.interface.schemas import UnknownRegion
from src.modules.risk_evaluator import INVALID_INPUT_ERROR
from src.modules.risk_evaluator import RiskEvaluator


# 创建启用状态的风险评估器
def build_evaluator(**overrides: object) -> RiskEvaluator:
    config = {
        "enabled": True,
        "intersection_ratio": 0.15,
        "near_margin_ratio": 0.03,
        "footprint_height_ratio": 0.25,
        "notice_distance_m": 30.0,
        "warning_distance_m": 15.0,
        "danger_distance_m": 6.0,
        "notice_ttc_s": 5.0,
        "warning_ttc_s": 3.0,
        "danger_ttc_s": 1.5,
        "track_iou_threshold": 0.30,
        "history_size": 5,
        "confirm_frames": 3,
        "release_frames": 3,
        "min_corridor_confidence": 0.45,
        "min_closing_speed_mps": 0.5,
        "max_track_gap_s": 1.5,
    }
    config.update(overrides)
    return RiskEvaluator(config)


# 创建测试用的矩形自车行驶走廊
def build_corridor(
    confidence: float = 0.9,
    enabled: bool = True,
) -> CorridorPredictionResult:
    corridor_mask = np.zeros((100, 100), dtype=np.uint8)
    corridor_mask[20:100, 35:65] = 255
    return CorridorPredictionResult(
        corridor_mask=corridor_mask,
        polygon=[
            (35, 99),
            (65, 99),
            (65, 20),
            (35, 20),
        ],
        centerline=[(50, 99), (50, 20)],
        confidence=confidence,
        inference_time_ms=1.0,
        error_code=0,
        error_message="OK",
        method="test",
        model_version="test",
        is_enabled=enabled,
    )


# 创建指定位置和距离的已知障碍物
def build_known_object(
    bbox: tuple = (40, 40, 60, 90),
    distance: float | None = 20.0,
) -> DetectedObject:
    return DetectedObject(
        class_name="vehicle",
        bbox=bbox,
        confidence=0.9,
        distance=distance,
    )


# 将二值掩码编码成项目使用的列优先游程格式
def encode_mask_rle(mask: np.ndarray) -> dict:
    flattened = np.ravel(mask > 0, order="F").astype(np.uint8)
    change_indices = np.flatnonzero(np.diff(flattened)) + 1
    run_starts = np.concatenate(
        (np.array([0]), change_indices)
    )
    run_ends = np.concatenate(
        (change_indices, np.array([flattened.size]))
    )
    counts: List[int] = (
        run_ends - run_starts
    ).tolist()
    if flattened.size > 0 and flattened[0] == 1:
        counts.insert(0, 0)
    return {
        "size": [mask.shape[0], mask.shape[1]],
        "counts": counts,
    }


# 测试走廊外障碍物不会产生业务风险
def test_outside_obstacle_is_safe() -> None:
    evaluator = build_evaluator()
    result = evaluator.evaluate(
        frame_id=0,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[
            build_known_object(
                bbox=(2, 40, 20, 90),
                distance=4.0,
            )
        ],
        unknown_regions=[],
    )

    assert result.is_valid
    assert result.risk_level == "safe"
    assert result.obstacle_risks[0].spatial_relation == "outside"


# 测试单帧近距离侵入只允许输出提示
def test_single_frame_close_conflict_is_notice() -> None:
    result = build_evaluator().evaluate(
        frame_id=0,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[build_known_object(distance=4.0)],
        unknown_regions=[],
    )

    assert result.risk_level == "notice"
    assert result.obstacle_risks[0].stable_frames == 1
    assert result.obstacle_risks[0].spatial_relation == "intersecting"


# 测试连续确认后的近距离侵入升级为危险
def test_stable_close_conflict_becomes_danger() -> None:
    evaluator = build_evaluator()
    results = []
    for frame_id in range(3):
        results.append(
            evaluator.evaluate(
                frame_id=frame_id,
                fps=10.0,
                corridor_result=build_corridor(),
                known_objects=[
                    build_known_object(distance=4.0)
                ],
                unknown_regions=[],
            )
        )

    assert results[0].risk_level == "notice"
    assert results[1].risk_level == "notice"
    assert results[2].risk_level == "danger"
    assert results[2].obstacle_risks[0].stable_frames == 3


# 测试未知障碍物优先使用真实掩码计算交叠
def test_unknown_region_uses_rle_mask() -> None:
    evaluator = build_evaluator(confirm_frames=1)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[70:90, 45:55] = 255
    unknown_region = UnknownRegion(
        bbox=(0, 0, 10, 10),
        score=0.9,
        distance=10.0,
        object_id="unstable-frame-id",
        area=200,
        mask_rle=encode_mask_rle(mask),
    )

    result = evaluator.evaluate(
        frame_id=0,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[unknown_region],
    )

    obstacle_risk = result.obstacle_risks[0]
    assert obstacle_risk.source == "unknown"
    assert obstacle_risk.spatial_relation == "intersecting"
    assert obstacle_risk.corridor_overlap == pytest.approx(1.0)
    assert obstacle_risk.risk_level == "warning"


# 测试距离未知的稳定侵入障碍物按保守规则预警
def test_missing_distance_conflict_is_warning() -> None:
    evaluator = build_evaluator()
    result = None
    for frame_id in range(3):
        result = evaluator.evaluate(
            frame_id=frame_id,
            fps=10.0,
            corridor_result=build_corridor(),
            known_objects=[
                build_known_object(distance=None)
            ],
            unknown_regions=[],
        )

    assert result is not None
    assert result.risk_level == "warning"
    assert result.obstacle_risks[0].ttc is None


# 测试视频距离变化可以生成相对TTC
def test_video_distance_history_generates_ttc() -> None:
    evaluator = build_evaluator(confirm_frames=1)
    distances = [20.0, 18.0, 16.0]
    result = None
    for frame_id, distance in zip(
        [0, 10, 20],
        distances,
    ):
        result = evaluator.evaluate(
            frame_id=frame_id,
            fps=10.0,
            corridor_result=build_corridor(),
            known_objects=[
                build_known_object(distance=distance)
            ],
            unknown_regions=[],
        )

    assert result is not None
    ttc = result.obstacle_risks[0].ttc
    assert ttc is not None
    assert ttc == pytest.approx(8.0)


# 测试风险降低需要连续安全帧释放
def test_risk_downgrade_uses_release_hysteresis() -> None:
    evaluator = build_evaluator(release_frames=3)
    for frame_id in range(3):
        danger_result = evaluator.evaluate(
            frame_id=frame_id,
            fps=10.0,
            corridor_result=build_corridor(),
            known_objects=[build_known_object(distance=4.0)],
            unknown_regions=[],
        )
    assert danger_result.risk_level == "danger"

    first_safe = evaluator.evaluate(
        frame_id=3,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[],
    )
    second_safe = evaluator.evaluate(
        frame_id=4,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[],
    )
    third_safe = evaluator.evaluate(
        frame_id=5,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[],
    )

    assert first_safe.risk_level == "danger"
    assert second_safe.risk_level == "danger"
    assert third_safe.risk_level == "safe"


# 测试走廊不可用时禁止输出虚假安全
def test_unavailable_corridor_is_not_safe() -> None:
    result = build_evaluator().evaluate(
        frame_id=0,
        fps=10.0,
        corridor_result=build_corridor(
            confidence=0.1,
        ),
        known_objects=[],
        unknown_regions=[],
    )

    assert not result.is_valid
    assert result.risk_level == "notice"
    assert result.system_status == SystemStatus.DEGRADED


# 测试无效视频FPS返回明确错误状态
def test_invalid_video_fps_returns_error() -> None:
    result = build_evaluator().evaluate(
        frame_id=0,
        fps=0.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[],
    )

    assert result.error_code == INVALID_INPUT_ERROR
    assert not result.is_valid
    assert result.risk_level != "safe"


# 测试风险模块可以在新视频开始前清空历史
def test_reset_clears_tracking_and_risk_state() -> None:
    evaluator = build_evaluator()
    for frame_id in range(3):
        evaluator.evaluate(
            frame_id=frame_id,
            fps=10.0,
            corridor_result=build_corridor(),
            known_objects=[build_known_object(distance=4.0)],
            unknown_regions=[],
        )
    evaluator.reset()

    result = evaluator.evaluate(
        frame_id=0,
        fps=10.0,
        corridor_result=build_corridor(),
        known_objects=[],
        unknown_regions=[],
    )

    assert result.risk_level == "safe"
    assert result.obstacle_risks == []
