"""
文件名: live_visualizer.py
用途: 按比例显示包含右侧风险信息面板的输出帧
作者: 张楚涵
创建日期: 2026-07-22
最后修改日期: 2026-08-17
"""

from types import TracebackType
from typing import Optional
from typing import Type

import cv2
import numpy as np


class LiveVisualizer:
    """负责在可调整大小的窗口中显示完整风险界面"""

    # 初始化窗口名称、退出按键和最大显示尺寸
    def __init__(
        self,
        window_name: str = "Road Risk Warning",
        exit_key: str = "q",
        delay: int = 1,
        max_window_width: int = 1920,
        max_window_height: int = 1080,
    ) -> None:
        if len(exit_key) != 1:
            raise ValueError("Exit key must contain one character.")
        if delay < 1:
            raise ValueError("Delay must be positive.")
        if max_window_width <= 0 or max_window_height <= 0:
            raise ValueError("Maximum window dimensions must be positive.")
        self.window_name = window_name
        self.exit_key = exit_key
        self.delay = delay
        self.max_window_width = max_window_width
        self.max_window_height = max_window_height
        self._closed = False
        self._initialized = False

    # 创建窗口并按输出帧比例设置显示尺寸
    def _initialize_window(self, frame: np.ndarray) -> None:
        frame_height, frame_width = frame.shape[:2]
        width_scale = self.max_window_width / frame_width
        height_scale = self.max_window_height / frame_height
        display_scale = min(1.0, width_scale, height_scale)
        display_width = max(
            1,
            int(round(frame_width * display_scale)),
        )
        display_height = max(
            1,
            int(round(frame_height * display_scale)),
        )
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )
        cv2.resizeWindow(
            self.window_name,
            display_width,
            display_height,
        )
        self._initialized = True

    # 显示一帧图像，返回是否继续运行
    def show(self, frame: np.ndarray) -> bool:
        if self._closed:
            return False
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must have shape H x W x 3.")
        if not self._initialized:
            self._initialize_window(frame)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.delay) & 0xFF
        if key in (ord(self.exit_key), 27):
            self.close()
            return False
        return True

    # 关闭显示窗口
    def close(self) -> None:
        if not self._closed:
            if self._initialized:
                cv2.destroyWindow(self.window_name)
            self._closed = True

    # 支持with语法
    def __enter__(self) -> "LiveVisualizer":
        return self

    # 退出with语句时关闭显示窗口
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()
