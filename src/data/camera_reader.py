"""
文件名: camera_reader.py
用途: 从本地摄像头读取实时图像帧
作者: 张楚涵
创建日期: 2026-07-22
最后修改日期: 2026-07-22
"""

from typing import Iterator
from typing import Optional
from typing import Tuple

import cv2
import numpy as np


class CameraReader:
    """负责从摄像头逐帧读取图像"""

    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.frame_id = 0

    # 打开摄像头
    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera.")
        if self.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    # 读取单帧图像
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if self.capture is None:
            raise RuntimeError("Camera is not opened.")
        success, frame = self.capture.read()
        if not success:
            return False, None, self.frame_id
        current_frame_id = self.frame_id
        self.frame_id += 1
        return True, frame, current_frame_id

    # 逐帧生成图像
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        while True:
            success, frame, frame_id = self.read()
            if not success or frame is None:
                break
            yield frame_id, frame

    # 获取摄像头信息
    def get_info(self) -> dict:
        if self.capture is None:
            raise RuntimeError("Camera is not opened.")
        return {
            "fps": self.capture.get(cv2.CAP_PROP_FPS),
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": 0,
        }

    # 释放摄像头
    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # 支持with语法
    def __enter__(self) -> "CameraReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()