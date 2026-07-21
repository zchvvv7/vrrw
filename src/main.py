"""
文件名: main.py
用途: 项目总流程
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-21
"""

import json
import logging
import os
import time

logging.disable(logging.CRITICAL)

import cv2
import yaml

from src.interface.schemas import FrameResult
from src.modules.road_segmenter import RoadSegmenter
from src.modules.unknown_detector import UnknownDetector
from src.utils.result_visualizer import draw_result


# 读取配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# 初始化运行日志
def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("road_risk_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    logging.getLogger("src.modules.road_segmenter").setLevel(logging.ERROR)
    return logger

# 生成单帧JSON记录
def build_frame_record(frame_id: int, road_result, unknown_regions: list) -> dict:
    return {
        "frame_id": frame_id,
        "road_segmentation": {
            "system_status": road_result.system_status,
            "global_confidence": road_result.global_confidence,
            "road_pixel_ratio": road_result.road_pixel_ratio,
            "error_code": road_result.error_code,
            "error_message": road_result.error_message,
            "inference_time_ms": road_result.inference_time_ms,
        },
        "unknown_regions": [
            {
                "bbox": list(region.bbox),
                "score": region.score,
                "distance": region.distance,
            }
            for region in unknown_regions
        ],
    }

# 运行道路风险视频可视化流程
def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    logger = setup_logger(config["output"]["log_path"])
    logger.info("Pipeline started.")
    logger.info("Input video: %s", config["input"]["video_path"])
    logger.info("Output video: %s", config["output"]["video_path"])
    logger.info("Output JSON: %s", config["output"]["json_path"])
    os.makedirs(os.path.dirname(config["output"]["video_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output"]["json_path"]), exist_ok=True)
    road_segmenter = RoadSegmenter(
        config_path=config["road_segmenter"]["config_path"],
    )
    unknown_detector = UnknownDetector(
        config=config["unknown_detector"],
    )
    capture = cv2.VideoCapture(config["input"]["video_path"])
    if not capture.isOpened():
        logger.error("Failed to read input video.")
        raise RuntimeError("Failed to read the input video, please check the path.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        config["output"]["video_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        logger.error("Failed to create output video.")
        raise RuntimeError("Failed to write the output video, please check the path.")
    frame_id = 0
    frame_skip = config["input"]["frame_skip"]
    frame_records = []
    start_time = time.time()
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue
        road_result = road_segmenter.predict(frame)
        unknown_regions = unknown_detector.predict(
            frame=frame,
            road_mask=road_result.mask,
            confidence_map=road_result.confidence_map,
        )
        result = FrameResult(
            frame_id=frame_id,
            road_mask=road_result.mask,
            known_objects=[],
            unknown_regions=unknown_regions,
            risk_level=road_result.system_status,
            major_reason=road_result.error_message,
        )
        output_frame = draw_result(frame, result)
        writer.write(output_frame)
        frame_records.append(
            build_frame_record(
                frame_id=frame_id,
                road_result=road_result,
                unknown_regions=unknown_regions,
            )
        )
        frame_id += 1
    capture.release()
    writer.release()
    elapsed_time = time.time() - start_time
    output_json = {
        "input_video": config["input"]["video_path"],
        "output_video": config["output"]["video_path"],
        "total_video_frames": total_frames,
        "total_processed_frames": len(frame_records),
        "elapsed_time_seconds": elapsed_time,
        "frames": frame_records,
    }
    with open(config["output"]["json_path"], "w", encoding="utf-8") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=2)
    logger.info("Pipeline finished.")
    logger.info("Processed frames: %s", len(frame_records))
    logger.info("Elapsed time seconds: %.2f", elapsed_time)


if __name__ == "__main__":
    run_pipeline("configs/default.yaml")