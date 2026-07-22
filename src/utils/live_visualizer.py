"""
文件名: live_visualizer.py
用途: 实时显示处理后的图像帧
作者: 张楚涵
创建日期: 2026-07-22
最后修改日期: 2026-07-22
"""

from typing import Optional

import cv2
import numpy as np


class LiveVisualizer:
    """负责实时显示可视化结果"""

    def __init__(
        self,
        window_name: str = "Road Risk Warning",
        exit_key: str = "q",
        delay: int = 1,
    ) -> None:
        self.window_name = window_name
        self.exit_key = exit_key
        self.delay = delay
        self._closed = False

    # 显示一帧图像，返回是否继续运行
    def show(self, frame: np.ndarray) -> bool:
        if self._closed:
            return False
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.delay) & 0xFF
        if key == ord(self.exit_key):
            self.close()
            return False
        return True

    # 关闭显示窗口
    def close(self) -> None:
        if not self._closed:
            cv2.destroyAllWindows()
            self._closed = True

    # 支持with语法
    def __enter__(self) -> "LiveVisualizer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()