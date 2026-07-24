"""
文件名: main.py
用途: 项目总流程
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-24
"""

import json
import logging
import os
import time
from typing import Tuple
from typing import Union

import cv2
import yaml

from src.data.camera_reader import CameraReader
from src.data.video_reader import VideoReader
from src.interface.schemas import FrameResult
from src.interface.schemas import RoadSegmentResult
from src.interface.schemas import UnknownDetectionResult
from src.modules.road_segmenter import RoadSegmenter
from src.modules.unknown_detector import UnknownDetector
from src.utils.live_visualizer import LiveVisualizer
from src.utils.result_visualizer import draw_result


# 读取配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# 初始化运行日志
def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(
        os.path.dirname(log_path),
        exist_ok=True,
    )
    logger = logging.getLogger("road_risk_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

# 创建输入源读取器
def build_reader(
    config: dict,
) -> Union[CameraReader, VideoReader]:
    source_type = config["input"]["source_type"]
    if source_type == "camera":
        return CameraReader(
            camera_id=config["input"]["camera_id"],
            width=config["input"]["width"],
            height=config["input"]["height"],
            fps=config["input"]["fps"],
        )
    if source_type == "video":
        return VideoReader(
            video_path=config["input"]["video_path"],
            frame_skip=config["input"]["frame_skip"],
        )
    raise ValueError(
        f"Unsupported source_type: {source_type}"
    )

# 根据模块状态生成当前帧风险显示信息
def build_frame_status(
    road_result: RoadSegmentResult,
    unknown_result: UnknownDetectionResult,
) -> Tuple[str, str]:
    if road_result.is_unavailable:
        return "unavailable", road_result.error_message
    if not unknown_result.is_successful:
        return "unavailable", unknown_result.error_message
    if road_result.is_degraded:
        return "degraded", road_result.error_message
    if unknown_result.regions:
        return "notice", "unknown_obstacle_detected"
    return "safe", "no_obstacle"

# 生成单帧JSON记录
def build_frame_record(
    frame_id: int,
    road_result: RoadSegmentResult,
    unknown_result: UnknownDetectionResult,
    risk_level: str,
    major_reason: str,
) -> dict:
    return {
        "frame_id": frame_id,
        "road_segmentation": {
            "system_status": road_result.system_status,
            "global_confidence": road_result.global_confidence,
            "road_pixel_ratio": road_result.road_pixel_ratio,
            "error_code": road_result.error_code,
            "error_message": road_result.error_message,
            "inference_time_ms": (
                road_result.inference_time_ms
            ),
        },
        "unknown_detection": {
            "model_version": unknown_result.model_version,
            "inference_time_ms": (
                unknown_result.inference_time_ms
            ),
            "error_code": unknown_result.error_code,
            "error_message": unknown_result.error_message,
            "region_count": len(unknown_result.regions),
        },
        "unknown_regions": [
            {
                "object_id": region.object_id,
                "bbox": list(region.bbox),
                "score": region.score,
                "area": region.area,
                "distance": region.distance,
                "mask_rle": region.mask_rle,
            }
            for region in unknown_result.regions
        ],
        "risk": {
            "risk_level": risk_level,
            "major_reason": major_reason,
        },
    }

# 保存JSON结果
def save_result_json(
    json_path: str,
    source_type: str,
    input_video: str,
    output_video: str,
    total_frames: int,
    frame_records: list,
    elapsed_time: float,
    interrupted: bool,
) -> None:
    output_json = {
        "source_type": source_type,
        "input_video": input_video,
        "output_video": output_video,
        "total_video_frames": total_frames,
        "total_processed_frames": len(frame_records),
        "elapsed_time_seconds": elapsed_time,
        "interrupted": interrupted,
        "frames": frame_records,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            output_json,
            file,
            ensure_ascii=False,
            indent=2,
        )

# 运行道路风险处理流程
def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    source_type = config["input"]["source_type"]
    logger = setup_logger(
        config["output"]["log_path"]
    )
    logger.info("Pipeline started.")
    logger.info("Source type: %s", source_type)
    logger.info(
        "Input video: %s",
        config["input"]["video_path"],
    )
    logger.info(
        "Output video: %s",
        config["output"]["video_path"],
    )
    logger.info(
        "Output JSON: %s",
        config["output"]["json_path"],
    )
    os.makedirs(
        os.path.dirname(
            config["output"]["video_path"]
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(
            config["output"]["json_path"]
        ),
        exist_ok=True,
    )
    road_segmenter = RoadSegmenter(
        config_path=(
            config["road_segmenter"]["config_path"]
        ),
    )
    unknown_detector = UnknownDetector(
        config=config["unknown_detector"],
    )
    reader = build_reader(config)
    writer = None
    live_visualizer = None
    frame_records = []
    total_frames = 0
    interrupted = False
    start_time = time.time()
    try:
        reader.open()
        source_info = reader.get_info()
        fps = source_info["fps"]
        width = source_info["width"]
        height = source_info["height"]
        total_frames = source_info["total_frames"]
        writer = cv2.VideoWriter(
            config["output"]["video_path"],
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            logger.error(
                "Failed to create output video."
            )
            raise RuntimeError(
                "Failed to write the output video. "
                "Please check the path."
            )
        if source_type == "camera":
            live_visualizer = LiveVisualizer(
                window_name="Road Risk Warning",
                exit_key="q",
                delay=1,
            )
        for frame_id, frame in reader.frames():
            road_result = road_segmenter.predict(frame)
            unknown_result = unknown_detector.predict(
                frame=frame,
                road_mask=road_result.mask,
            )
            risk_level, major_reason = build_frame_status(
                road_result,
                unknown_result,
            )
            if not unknown_result.is_successful:
                logger.error(
                    "Unknown detection failed: %s",
                    unknown_result.error_message,
                )
            result = FrameResult(
                frame_id=frame_id,
                road_mask=road_result.mask,
                known_objects=[],
                unknown_regions=unknown_result.regions,
                risk_level=risk_level,
                major_reason=major_reason,
                anomaly_mask=(
                    unknown_result.anomaly_mask
                ),
            )
            output_frame = draw_result(frame, result)
            writer.write(output_frame)

            if live_visualizer is not None:
                should_continue = live_visualizer.show(
                    output_frame
                )
                if not should_continue:
                    interrupted = True
                    break
            frame_records.append(
                build_frame_record(
                    frame_id=frame_id,
                    road_result=road_result,
                    unknown_result=unknown_result,
                    risk_level=risk_level,
                    major_reason=major_reason,
                )
            )
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Pipeline interrupted by user.")
    finally:
        reader.release()
        if live_visualizer is not None:
            live_visualizer.close()
        if writer is not None:
            writer.release()
        elapsed_time = time.time() - start_time
        save_result_json(
            json_path=config["output"]["json_path"],
            source_type=source_type,
            input_video=config["input"]["video_path"],
            output_video=config["output"]["video_path"],
            total_frames=total_frames,
            frame_records=frame_records,
            elapsed_time=elapsed_time,
            interrupted=interrupted,
        )
        if interrupted:
            logger.info(
                "Result saved after interruption."
            )
        else:
            logger.info("Pipeline finished.")
        logger.info(
            "Processed frames: %s",
            len(frame_records),
        )
        logger.info(
            "Elapsed time seconds: %.2f",
            elapsed_time,
        )

if __name__ == "__main__":
    run_pipeline("configs/default.yaml")
