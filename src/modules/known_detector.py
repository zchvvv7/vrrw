"""
文件名: known_detector.py
用途: 检测已知类型道路障碍物
作者:周子懿
创建日期: 2026-07-16
最后修改日期: 2026-07-24
"""

from typing import List
from typing import Set

import numpy as np
from ultralytics import YOLO

from src.interface.schemas import DetectedObject


class KnownDetector:
    """负责检测锥桶、护栏、坑洞、车辆等已知障碍物"""

    # 初始化已知障碍物检测器并加载YOLO模型
    def __init__(self, config: dict) -> None:
        self._model_name = config.get("model_name", "yolo11n.pt")
        self._conf_threshold = float(
            config.get("conf_threshold", 0.25)
        )
        self._iou_threshold = float(
            config.get("iou_threshold", 0.45)
        )
        self._device = config.get("device", "cpu")
        class_list = config.get("classes", [])
        self._target_classes: Set[str] = {
            name.lower().strip() for name in class_list
        }
        self._model = YOLO(self._model_name)

    # 对单帧图像执行已知障碍物检测
    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        if frame is None or frame.size == 0:
            return []

        results = self._model.predict(
            source=frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
        )

        detected_objects: List[DetectedObject] = []
        if not results:
            return detected_objects

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detected_objects

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id]).lower()
            if (
                self._target_classes
                and class_name not in self._target_classes
            ):
                continue

            xyxy = box.xyxy[0].tolist()
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            confidence = float(box.conf[0].item())
            detected_objects.append(
                DetectedObject(
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    distance=None,
                )
            )

        return detected_objects