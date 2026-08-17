"""
文件名: test_road_segmenter.py
用途: 测试RoadSegmenter模块
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-27
"""

from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from src.config.config import RoadSegmenterConfig
from src.modules import road_segmenter as road_segmenter_module
from src.modules.road_segmenter import RoadSegmenter


class FakeSegformerProcessor:
    """模拟SegFormer图像处理器"""

    # 将输入图像转换为固定尺寸测试张量
    def __call__(
        self,
        images: np.ndarray,
        return_tensors: str,
    ) -> dict:
        return {
            "pixel_values": torch.zeros(
                (1, 3, 64, 64),
                dtype=torch.float32,
            )
        }


class FakeProcessorFactory:
    """模拟SegFormer图像处理器工厂"""

    # 返回离线测试图像处理器
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
    ) -> FakeSegformerProcessor:
        return FakeSegformerProcessor()


class FakeSegformerModel:
    """模拟可重复输出的SegFormer模型"""

    # 初始化模拟模型类别数量
    def __init__(self, num_labels: int) -> None:
        self._num_labels = num_labels

    # 模拟将模型移动到目标设备
    def to(
        self,
        device: torch.device,
    ) -> "FakeSegformerModel":
        return self

    # 模拟切换模型为推理模式
    def eval(self) -> "FakeSegformerModel":
        return self

    # 生成道路类别占优的固定分割输出
    def __call__(self, **inputs: Any) -> SimpleNamespace:
        batch_size = inputs["pixel_values"].shape[0]
        logits = torch.zeros(
            (
                batch_size,
                self._num_labels,
                64,
                64,
            ),
            dtype=torch.float32,
        )
        logits[:, 0, :, :] = 5.0
        return SimpleNamespace(logits=logits)

    # 模拟加载本地模型参数
    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool,
    ) -> tuple:
        return [], []


class FakeModelFactory:
    """模拟SegFormer模型工厂"""

    # 返回离线测试模型
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int,
    ) -> FakeSegformerModel:
        return FakeSegformerModel(num_labels)


# 模拟本地权重加载以避免读取真实大模型
def fake_load_checkpoint(
    segmenter: RoadSegmenter,
    checkpoint_path: str,
) -> None:
    return None


# 将道路分割外部模型替换为离线测试实现
@pytest.fixture(autouse=True)
def mock_segformer_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        road_segmenter_module,
        "SegformerImageProcessor",
        FakeProcessorFactory,
    )
    monkeypatch.setattr(
        road_segmenter_module,
        "SegformerForSemanticSegmentation",
        FakeModelFactory,
    )
    monkeypatch.setattr(
        RoadSegmenter,
        "load_checkpoint",
        fake_load_checkpoint,
    )


class TestRoadSegmenter:
    """RoadSegmenter模块测试类"""

    # 创建RoadSegmenter实例（禁用质量检测以避免随机噪声影响）
    @pytest.fixture
    def segmenter(self) -> RoadSegmenter:
        config = RoadSegmenterConfig()
        config.quality.enable_quality_check = False
        return RoadSegmenter(config=config)

    # 创建随机测试图像（480x640x3）
    @pytest.fixture
    def test_frame(self) -> np.ndarray:
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # 创建灰度测试图像（转换为BGR格式）
    @pytest.fixture
    def gray_frame(self) -> np.ndarray:
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 创建默认配置对象
    @pytest.fixture
    def config(self) -> RoadSegmenterConfig:
        return RoadSegmenterConfig()

    # 测试初始化是否成功
    def test_init(self, segmenter: RoadSegmenter) -> None:
        assert segmenter is not None
        assert segmenter.device is not None
        assert segmenter.model is not None
        assert segmenter.processor is not None
        assert segmenter._model_initialized is True

    # 测试从配置文件初始化
    def test_init_with_config_path(self) -> None:
        segmenter = RoadSegmenter(config_path="configs/road_segmenter.yaml")
        assert segmenter is not None
        assert "segformer" in segmenter.config.model.model_name.lower()

    # 测试使用自定义配置初始化
    def test_init_with_custom_config(self) -> None:
        config = RoadSegmenterConfig()
        config.labels.dataset = "bdd100k"
        segmenter = RoadSegmenter(config=config)
        assert segmenter.config.get_road_label() == 2

    # 测试获取模型信息
    def test_get_model_info(self, segmenter: RoadSegmenter) -> None:
        info = segmenter.get_model_info()
        assert "model_name" in info
        assert "num_classes" in info
        assert "dataset" in info
        assert "road_label" in info
        assert "device" in info
        assert "model_initialized" in info

    # 测试预测输出结构是否完整
    def test_predict_output_structure(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert hasattr(result, "mask")
        assert hasattr(result, "boundary")
        assert hasattr(result, "confidence_map")
        assert hasattr(result, "global_confidence")
        assert hasattr(result, "road_pixel_ratio")
        assert hasattr(result, "error_code")
        assert hasattr(result, "error_message")
        assert hasattr(result, "inference_time_ms")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "is_successful")

    # 测试预测输出掩码尺寸是否正确
    def test_predict_output_shape(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert result.mask.shape == (480, 640)

    # 测试预测输出掩码数据类型是否正确
    def test_predict_output_dtype(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert result.mask.dtype == np.uint8

    # 测试预测输出掩码值是否为二值（0或255）
    def test_predict_output_values(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        unique_values = np.unique(result.mask)
        for val in unique_values:
            assert val in [0, 255], (
                f"Mask should only contain 0 or 255, got {unique_values}"
            )

    # 测试边界输出是否正确
    def test_predict_boundary_output(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        if result.boundary is not None:
            assert result.boundary.shape == (480, 640)
            assert result.boundary.dtype == np.uint8

    # 测试置信度图输出是否正确
    def test_predict_confidence_map_output(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        if result.confidence_map is not None:
            assert result.confidence_map.shape == (480, 640)
            assert np.all(result.confidence_map >= 0) and np.all(
                result.confidence_map <= 1
            )

    # 测试全局置信度范围是否正确（0-1）
    def test_predict_global_confidence_range(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert 0.0 <= result.global_confidence <= 1.0

    # 测试道路像素比例范围是否正确（0-1）
    def test_predict_road_pixel_ratio(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert 0.0 <= result.road_pixel_ratio <= 1.0

    # 测试推理时间是否非负
    def test_predict_inference_time(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert result.inference_time_ms >= 0

    # 测试正常预测是否成功
    def test_predict_success(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        assert result.is_successful is True
        assert result.error_code == 0
        assert result.error_message == "OK"

    # 测试灰度图像（转换为BGR）预测
    def test_predict_gray_frame(
        self,
        segmenter: RoadSegmenter,
        gray_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(gray_frame)
        assert result.mask.shape == (480, 640)
        assert result.mask.dtype == np.uint8

    # 测试不同尺寸图像的预测
    def test_predict_different_sizes(
        self,
        segmenter: RoadSegmenter,
    ) -> None:
        sizes = [(360, 640), (720, 1280), (240, 320)]
        for height, width in sizes:
            frame = np.random.randint(
                0, 256, (height, width, 3), dtype=np.uint8
            )
            result = segmenter.predict(frame)
            assert result.mask.shape == (height, width)

    # 测试相同输入的预测结果是否一致
    def test_predict_consistency(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result1 = segmenter.predict(test_frame.copy())
        result2 = segmenter.predict(test_frame.copy())
        assert np.array_equal(result1.mask, result2.mask)
        assert result1.error_code == result2.error_code

    # 测试带时间戳的预测
    def test_predict_with_timestamp(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        timestamp = 1234567890.123
        result = segmenter.predict(test_frame, timestamp=timestamp)
        assert result.timestamp == timestamp

    # 测试None输入的处理
    def test_predict_none_input(self, segmenter: RoadSegmenter) -> None:
        result = segmenter.predict(None)
        assert result.is_successful is False
        assert result.error_code == -1
        assert "None" in result.error_message

    # 测试空帧输入的处理
    def test_predict_empty_frame(self, segmenter: RoadSegmenter) -> None:
        empty_frame = np.array([], dtype=np.uint8)
        result = segmenter.predict(empty_frame)
        assert result.is_successful is False
        assert result.error_code == -1

    # 测试无效形状输入（缺少通道维度）的处理
    def test_predict_invalid_shape(self, segmenter: RoadSegmenter) -> None:
        invalid_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        result = segmenter.predict(invalid_frame)
        assert result.is_successful is False
        assert result.error_code == -1

    # 测试单通道输入的处理
    def test_predict_single_channel(
        self,
        segmenter: RoadSegmenter,
    ) -> None:
        single_channel = np.random.randint(
            0, 256, (480, 640, 1), dtype=np.uint8
        )
        result = segmenter.predict(single_channel)
        assert result.is_successful is False
        assert result.error_code == -1

    # 测试转换为字典的功能
    def test_predict_to_dict(
        self,
        segmenter: RoadSegmenter,
        test_frame: np.ndarray,
    ) -> None:
        result = segmenter.predict(test_frame)
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "mask" in result_dict
        assert "boundary" in result_dict
        assert "confidence_map" in result_dict
        assert "global_confidence" in result_dict
        assert "error_code" in result_dict

    # 测试不同数据集的标签映射
    def test_dataset_label_mapping(self) -> None:
        datasets = ["cityscapes", "bdd100k", "mapillary"]
        expected_labels = [0, 2, 20]
        for dataset, expected_label in zip(datasets, expected_labels):
            config = RoadSegmenterConfig()
            config.labels.dataset = dataset
            segmenter = RoadSegmenter(config=config)
            assert segmenter.config.get_road_label() == expected_label

    # 测试边界检测禁用的情况
    def test_boundary_detection_disabled(
        self,
        test_frame: np.ndarray,
    ) -> None:
        config = RoadSegmenterConfig()
        config.post_processing.boundary_detection = False
        segmenter = RoadSegmenter(config=config)
        result = segmenter.predict(test_frame)
        assert result.boundary is None

    # 测试掩码平滑禁用的情况
    def test_mask_smoothing_disabled(
        self,
        test_frame: np.ndarray,
    ) -> None:
        config = RoadSegmenterConfig()
        config.post_processing.mask_smoothing = False
        segmenter = RoadSegmenter(config=config)
        result = segmenter.predict(test_frame)
        assert result.mask is not None

    # 测试load_checkpoint方法是否存在
    def test_load_checkpoint_method_exists(
        self,
        segmenter: RoadSegmenter,
    ) -> None:
        assert hasattr(segmenter, "load_checkpoint")

    # 测试合成道路图像的预测
    def test_predict_synthetic_road_image(
        self,
        segmenter: RoadSegmenter,
    ) -> None:
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[: int(height * 0.6), :] = [128, 128, 128]
        frame[int(height * 0.6) :, :] = [200, 200, 200]
        result = segmenter.predict(frame)
        assert result.is_successful
        road_pixels = np.sum(result.mask == 255)
        road_ratio = road_pixels / (height * width)
        assert 0.0 <= road_ratio <= 1.0, (
            f"Invalid road pixel ratio: {road_ratio}"
        )

    # 测试轮廓平滑启用的情况
    def test_contour_smoothing_enabled(
        self,
        test_frame: np.ndarray,
    ) -> None:
        config = RoadSegmenterConfig()
        config.post_processing.contour_smoothing = True
        segmenter = RoadSegmenter(config=config)
        result = segmenter.predict(test_frame)
        assert result.mask is not None
        assert result.mask.shape == (480, 640)
        assert result.mask.dtype == np.uint8

    # 测试轮廓平滑禁用的情况
    def test_contour_smoothing_disabled(
        self,
        test_frame: np.ndarray,
    ) -> None:
        config = RoadSegmenterConfig()
        config.post_processing.contour_smoothing = False
        segmenter = RoadSegmenter(config=config)
        result = segmenter.predict(test_frame)
        assert result.mask is not None
        assert result.mask.shape == (480, 640)

    # 测试轮廓平滑对mask边缘的影响
    def test_contour_smoothing_effect(self) -> None:
        height, width = 480, 640
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (540, 380), 255, -1)
        cv2.rectangle(mask, (200, 200), (440, 280), 0, -1)

        config = RoadSegmenterConfig()
        config.post_processing.mask_smoothing = True
        config.post_processing.contour_smoothing = True
        segmenter = RoadSegmenter(config=config)

        smoothed_mask = segmenter._smooth_mask(mask.copy())
        assert smoothed_mask.shape == mask.shape
        assert smoothed_mask.dtype == np.uint8

        contours_original, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_smoothed, _ = cv2.findContours(
            smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours_original) > 0
        assert len(contours_smoothed) > 0

    # 测试轮廓平滑的配置参数是否正确加载
    def test_contour_smoothing_config(self) -> None:
        config = RoadSegmenterConfig()
        config.post_processing.contour_smoothing = True
        config.post_processing.smoothing_factor = 10.0
        config.post_processing.spline_order = 3
        config.post_processing.resample_points = 300

        segmenter = RoadSegmenter(config=config)
        assert segmenter.config.post_processing.contour_smoothing is True
        assert segmenter.config.post_processing.smoothing_factor == 10.0
        assert segmenter.config.post_processing.spline_order == 3
        assert segmenter.config.post_processing.resample_points == 300
