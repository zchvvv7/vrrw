"""
文件名: test_main_unknown_integration.py
用途: 测试已知和未知障碍物检测结果与主流程的数据衔接
作者: 张楚涵
创建日期: 2026-07-24
最后修改日期: 2026-08-03
"""

import numpy as np

from src.interface.schemas import CorridorPredictionResult
from src.interface.schemas import DetectedObject
from src.interface.schemas import DistanceEstimationResult
from src.interface.schemas import KnownDetectionResult
from src.interface.schemas import ObstacleRisk
from src.interface.schemas import RiskEvaluationResult
from src.interface.schemas import RoadSegmentResult
from src.interface.schemas import SystemStatus
from src.interface.schemas import UnknownDetectionResult
from src.interface.schemas import UnknownRegion
from src.main import build_frame_record
from src.main import build_risk_system_status


# 创建测试道路分割结果
def build_road_result() -> RoadSegmentResult:
    return RoadSegmentResult(
        mask=np.full(
            (20, 40),
            255,
            dtype=np.uint8,
        ),
        boundary=None,
        confidence_map=None,
        global_confidence=0.9,
        road_pixel_ratio=0.8,
        error_code=0,
        error_message="OK",
        inference_time_ms=10.0,
        system_status=SystemStatus.NORMAL,
    )


# 创建无目标的测试已知障碍物检测结果
def build_empty_known_result() -> KnownDetectionResult:
    return KnownDetectionResult(
        objects=[],
        inference_time_ms=5.0,
        error_code=0,
        error_message="OK",
        model_version="yolo-test",
    )


# 创建包含目标的测试已知障碍物检测结果
def build_known_result() -> KnownDetectionResult:
    detected_object = DetectedObject(
        class_name="cone",
        bbox=(5, 8, 12, 18),
        confidence=0.85,
    )
    return KnownDetectionResult(
        objects=[detected_object],
        inference_time_ms=5.0,
        error_code=0,
        error_message="OK",
        model_version="yolo-test",
    )


# 创建测试未知障碍物检测结果
def build_unknown_result() -> UnknownDetectionResult:
    region = UnknownRegion(
        bbox=(10, 10, 20, 18),
        score=0.9,
        object_id="unknown-000",
        area=80,
    )
    return UnknownDetectionResult(
        score_map=np.zeros(
            (20, 40),
            dtype=np.float32,
        ),
        anomaly_mask=np.zeros(
            (20, 40),
            dtype=np.uint8,
        ),
        regions=[region],
        inference_time_ms=20.0,
        error_code=0,
        error_message="OK",
        model_version="mask2anomaly-test",
    )


# 测试未知障碍物结果可以写入单帧记录
def test_unknown_result_is_written_to_frame_record() -> None:
    road_result = build_road_result()
    known_result = build_empty_known_result()
    unknown_result = build_unknown_result()
    frame_record = build_frame_record(
        frame_id=1,
        road_result=road_result,
        known_result=known_result,
        unknown_result=unknown_result,
        risk_level="notice",
        major_reason="unknown_obstacle_detected",
    )

    assert frame_record["unknown_detection"]["error_code"] == 0
    assert frame_record["unknown_detection"]["region_count"] == 1
    assert frame_record["unknown_regions"][0]["score"] == 0.9
    assert frame_record["risk"]["risk_level"] == "notice"


# 测试已知障碍物结果可以写入单帧记录
def test_known_result_is_written_to_frame_record() -> None:
    road_result = build_road_result()
    known_result = build_known_result()
    unknown_result = build_unknown_result()
    frame_record = build_frame_record(
        frame_id=1,
        road_result=road_result,
        known_result=known_result,
        unknown_result=unknown_result,
        risk_level="notice",
        major_reason="known_obstacle_detected",
    )

    assert frame_record["known_detection"]["error_code"] == 0
    assert frame_record["known_detection"]["object_count"] == 1
    assert frame_record["known_objects"][0]["class_name"] == "cone"
    assert frame_record["known_objects"][0]["confidence"] == 0.85


# 测试距离和走廊接口结果可以写入单帧记录
def test_new_module_results_are_written_to_record() -> None:
    road_result = build_road_result()
    known_result = build_known_result()
    unknown_result = build_unknown_result()
    known_result.objects[0].distance = 8.5
    distance_result = DistanceEstimationResult(
        known_objects=known_result.objects,
        inference_time_ms=4.0,
        error_code=0,
        error_message="OK",
        method="test_distance",
        model_version="test-v1",
        is_enabled=True,
    )
    corridor_result = CorridorPredictionResult(
        corridor_mask=np.zeros(
            (20, 40),
            dtype=np.uint8,
        ),
        polygon=[
            (5, 19),
            (35, 19),
            (25, 5),
            (15, 5),
        ],
        centerline=[
            (20, 19),
            (20, 5),
        ],
        confidence=0.8,
        inference_time_ms=2.0,
        error_code=0,
        error_message="OK",
        method="test_corridor",
        model_version="test-v1",
        is_enabled=True,
    )

    frame_record = build_frame_record(
        frame_id=1,
        road_result=road_result,
        known_result=known_result,
        unknown_result=unknown_result,
        risk_level="notice",
        major_reason="known_obstacle_detected",
        distance_result=distance_result,
        corridor_result=corridor_result,
    )

    assert frame_record["known_objects"][0]["distance"] == 8.5
    assert frame_record["distance_estimation"]["enabled"]
    assert frame_record["distance_estimation"]["method"] == "test_distance"
    assert frame_record["corridor_prediction"]["confidence"] == 0.8
    assert frame_record["corridor_prediction"]["polygon"][0] == [
        5,
        19,
    ]


# 测试视频风险结果可以完整写入单帧JSON记录
def test_video_risk_result_is_written_to_record() -> None:
    road_result = build_road_result()
    known_result = build_known_result()
    unknown_result = build_unknown_result()
    obstacle_risk = ObstacleRisk(
        object_id="known-track-0001",
        source="known",
        class_name="cone",
        bbox=(5, 8, 12, 18),
        distance=5.0,
        corridor_overlap=0.8,
        spatial_relation="intersecting",
        ttc=1.2,
        risk_level="danger",
        major_reason="known_obstacle_intersecting_critical",
        stable_frames=3,
    )
    risk_result = RiskEvaluationResult(
        risk_level="danger",
        major_reason="known_obstacle_intersecting_critical",
        obstacle_risks=[obstacle_risk],
        system_status=SystemStatus.NORMAL,
        is_valid=True,
        inference_time_ms=0.5,
        error_code=0,
        error_message="OK",
        model_version="video-risk-test",
        is_enabled=True,
    )

    frame_record = build_frame_record(
        frame_id=1,
        road_result=road_result,
        known_result=known_result,
        unknown_result=unknown_result,
        risk_level=risk_result.risk_level,
        major_reason=risk_result.major_reason,
        risk_result=risk_result,
    )

    assert frame_record["risk"]["is_valid"]
    assert frame_record["risk"]["risk_level"] == "danger"
    assert frame_record["risk"]["ttc"] == 1.2
    assert frame_record["risk"]["corridor_overlap"] == 0.8
    assert frame_record["risk"]["obstacles"][0]["ttc"] == 1.2
    assert (
        frame_record["risk"]["obstacles"][0]["spatial_relation"]
        == "intersecting"
    )


# 测试感知模块正常时风险系统状态可用
def test_risk_system_status_is_normal() -> None:
    road_result = build_road_result()
    known_result = build_known_result()
    unknown_result = build_unknown_result()
    distance_result = DistanceEstimationResult(
        known_objects=known_result.objects,
        inference_time_ms=1.0,
        error_code=0,
        error_message="OK",
        method="test",
        model_version="test",
        is_enabled=True,
    )
    corridor_result = CorridorPredictionResult(
        corridor_mask=np.full(
            (20, 40),
            255,
            dtype=np.uint8,
        ),
        polygon=[],
        centerline=[],
        confidence=0.9,
        inference_time_ms=1.0,
        error_code=0,
        error_message="OK",
        method="test",
        model_version="test",
        is_enabled=True,
    )

    system_status = build_risk_system_status(
        road_result=road_result,
        known_result=known_result,
        unknown_result=unknown_result,
        distance_result=distance_result,
        corridor_result=corridor_result,
    )

    assert system_status == SystemStatus.NORMAL
