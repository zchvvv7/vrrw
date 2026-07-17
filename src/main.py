"""
文件名: main.py
用途: 读取视频，逐帧调用道路分割模块，并输出可行驶区域可视化视频
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-17
"""

import yaml
import logging

import cv2

from src.interface.schemas import FrameResult
from src.modules.road_segmenter import RoadSegmenter
from src.utils.result_visualizer import draw_result


# 读取配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# 运行道路分割视频可视化流程
def run_pipeline(config_path: str) -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("src.modules.road_segmenter").setLevel(logging.ERROR)
    config = load_config(config_path)
    road_segmenter = RoadSegmenter(
        config_path=config["road_segmenter"]["config_path"],
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
        road_result = road_segmenter.predict(frame)
        result = FrameResult(
            frame_id=frame_id,
            road_mask=road_result.mask,
            known_objects=[],
            unknown_regions=[],
            risk_level=road_result.system_status,
            major_reason=road_result.error_message,
        )
        output_frame = draw_result(frame, result)
        writer.write(output_frame)
        frame_id += 1

    capture.release()
    writer.release()


if __name__ == "__main__":
    run_pipeline("configs/default.yaml")