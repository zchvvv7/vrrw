"""
文件名: test_known_detector.py
用途: 测试已知障碍物检测模块的结果转换和错误处理
作者: 张楚涵
创建日期: 2026-07-27
最后修改日期: 2026-07-27
"""

from typing import Any

import numpy as np

from src.modules.known_detector import INFERENCE_ERROR
from src.modules.known_detector import INVALID_INPUT_ERROR
from src.modules.known_detector import MODEL_LOAD_ERROR
from src.modules.known_detector import SUCCESS
from src.modules.known_detector import KnownDetector


class FakeBox:
    """模拟单个YOLO检测框"""

    # 初始化模拟检测框数据
    def __init__(
        self,
        class_id: int,
        confidence: float,
        bbox: list,
    ) -> None:
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([bbox])


class FakeResult:
    """模拟单帧YOLO推理结果"""

    # 初始化模拟类别和检测框
    def __init__(self) -> None:
        self.names = {
            0: "cone",
            1: "vehicle",
        }
        self.boxes = [
            FakeBox(
                class_id=0,
                confidence=0.9,
                bbox=[10.0, 20.0, 30.0, 40.0],
            ),
            FakeBox(
                class_id=1,
                confidence=0.8,
                bbox=[50.0, 60.0, 70.0, 80.0],
            ),
        ]


class FakeModel:
    """模拟正常工作的YOLO模型"""

    # 初始化模拟模型类别
    def __init__(self) -> None:
        self.names = {
            0: "cone",
            1: "vehicle",
        }

    # 返回模拟YOLO推理结果
    def predict(self, **kwargs: Any) -> list:
        return [FakeResult()]


class FailingModel:
    """模拟推理失败的YOLO模型"""

    # 初始化模拟模型类别
    def __init__(self) -> None:
        self.names = {
            0: "cone",
        }

    # 抛出模拟推理异常
    def predict(self, **kwargs: Any) -> list:
        raise RuntimeError("simulated inference failure")


# 创建已知障碍物检测测试配置
def build_config() -> dict:
    return {
        "model_name": "checkpoints/yolo_best.pt",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "device": "cpu",
        "classes": [
            "cone",
        ],
    }


# 测试YOLO结果可以转换为项目检测对象
def test_predict_converts_yolo_result() -> None:
    detector = KnownDetector(
        config=build_config(),
        model=FakeModel(),
    )
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    result = detector.predict(frame)

    assert result.error_code == SUCCESS
    assert result.is_successful
    assert len(result.objects) == 1
    assert result.objects[0].class_name == "cone"
    assert result.objects[0].bbox == (
        10,
        20,
        30,
        40,
    )
    assert result.objects[0].confidence == 0.9


# 测试无效输入会返回明确错误码
def test_invalid_frame_returns_error() -> None:
    detector = KnownDetector(
        config=build_config(),
        model=FakeModel(),
    )
    frame = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    result = detector.predict(frame)

    assert result.error_code == INVALID_INPUT_ERROR
    assert result.objects == []


# 测试推理异常会转换为失败结果
def test_inference_failure_returns_error() -> None:
    detector = KnownDetector(
        config=build_config(),
        model=FailingModel(),
    )
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    result = detector.predict(frame)

    assert result.error_code == INFERENCE_ERROR
    assert result.objects == []
    assert "simulated inference failure" in result.error_message


# 测试模型文件缺失会转换为加载失败结果
def test_missing_model_returns_error() -> None:
    config = build_config()
    config["model_name"] = "checkpoints/missing_yolo.pt"
    detector = KnownDetector(config=config)
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    result = detector.predict(frame)

    assert result.error_code == MODEL_LOAD_ERROR
    assert result.objects == []
    assert "FileNotFoundError" in result.error_message
