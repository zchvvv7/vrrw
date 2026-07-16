"""
文件名: main.py
用途: 项目入口，读取视频并输出风险结果视频
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import yaml

import cv2

from src.interface.schemas import FrameResult
from src.modules.known_detector import KnownDetector
from src.modules.risk_evaluator import RiskEvaluator
from src.modules.road_segmenter import RoadSegmenter
from src.modules.unknown_detector import UnknownDetector
from src.utils.result_visualizer import draw_result


# 读取配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 运行视频处理主流程
def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    road_segmenter = RoadSegmenter()
    known_detector = KnownDetector()
    unknown_detector = UnknownDetector()
    risk_evaluator = RiskEvaluator(
        warning_distance=config["risk"]["warning_distance"],
        danger_distance=config["risk"]["danger_distance"],
    )

    capture = cv2.VideoCapture(config["input"]["video_path"])
    if not capture.isOpened():
        raise RuntimeError("Failed to read the input video, please check the path.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        config["output"]["video_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to write the output video, please check the path.")
    frame_id = 0
    frame_skip = config["input"]["frame_skip"]

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue
        road_mask = road_segmenter.predict(frame)
        known_objects = known_detector.predict(frame)
        unknown_regions = unknown_detector.predict(frame)
        risk_level, major_reason = risk_evaluator.evaluate(
            known_objects,
            unknown_regions,
        )
        result = FrameResult(
            frame_id=frame_id,
            road_mask=road_mask,
            known_objects=known_objects,
            unknown_regions=unknown_regions,
            risk_level=risk_level,
            major_reason=major_reason,
        )
        output_frame = draw_result(frame, result)
        writer.write(output_frame)
        frame_id += 1

    capture.release()
    writer.release()


if __name__ == "__main__":
    run_pipeline("configs/default.yaml")