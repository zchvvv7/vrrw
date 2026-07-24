"""
文件名: test_main_unknown_integration.py
用途: 测试未知障碍物检测结果与主流程的数据衔接
作者: 张楚涵
创建日期: 2026-07-24
最后修改日期: 2026-07-24
"""

import numpy as np

from src.interface.schemas import RoadSegmentResult
from src.interface.schemas import SystemStatus
from src.interface.schemas import UnknownDetectionResult
from src.interface.schemas import UnknownRegion
from src.main import build_frame_record
from src.main import build_frame_status


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


# 测试未知障碍物会触发notice状态
def test_unknown_region_sets_notice_status() -> None:
    road_result = build_road_result()
    unknown_result = build_unknown_result()

    risk_level, major_reason = build_frame_status(
        road_result,
        unknown_result,
    )

    assert risk_level == "notice"
    assert major_reason == "unknown_obstacle_detected"


# 测试未知障碍物结果可以写入单帧记录
def test_unknown_result_is_written_to_frame_record() -> None:
    road_result = build_road_result()
    unknown_result = build_unknown_result()
    frame_record = build_frame_record(
        frame_id=1,
        road_result=road_result,
        unknown_result=unknown_result,
        risk_level="notice",
        major_reason="unknown_obstacle_detected",
    )

    assert frame_record["unknown_detection"]["error_code"] == 0
    assert frame_record["unknown_detection"]["region_count"] == 1
    assert frame_record["unknown_regions"][0]["score"] == 0.9
    assert frame_record["risk"]["risk_level"] == "notice"
