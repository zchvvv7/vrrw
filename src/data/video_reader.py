"""
文件名: video_reader.py
用途: 从视频文件逐帧读取图像
作者: 张楚涵
创建日期: 2026-07-22
最后修改日期: 2026-07-22
"""

from typing import Iterator
from typing import Optional
from typing import Tuple

import cv2
import numpy as np


class VideoReader:
    """负责从视频文件逐帧读取图像"""

    def __init__(self, video_path: str, frame_skip: int = 1) -> None:
        self.video_path = video_path
        self.frame_skip = max(frame_skip, 1)
        self.capture = None
        self.frame_id = 0

    # 打开视频文件
    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open input video.")

    # 读取单帧图像
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if self.capture is None:
            raise RuntimeError("Video is not opened.")
        while True:
            success, frame = self.capture.read()
            if not success:
                return False, None, self.frame_id
            current_frame_id = self.frame_id
            self.frame_id += 1
            if current_frame_id % self.frame_skip == 0:
                return True, frame, current_frame_id

    # 逐帧生成图像
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        while True:
            success, frame, frame_id = self.read()
            if not success or frame is None:
                break
            yield frame_id, frame

    # 获取视频信息
    def get_info(self) -> dict:
        if self.capture is None:
            raise RuntimeError("Video is not opened.")
        return {
            "fps": self.capture.get(cv2.CAP_PROP_FPS),
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    # 释放视频资源
    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # 支持with语法
    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()