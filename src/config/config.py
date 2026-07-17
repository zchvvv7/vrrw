"""
文件名: config.py
用途: 配置管理类，支持YAML配置加载和参数管理
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-17
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """SegFormer模型配置"""
    model_name: str = (
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    )
    checkpoint_path: Optional[str] = None
    num_classes: int = 19


@dataclass
class DataConfig:
    """数据配置类"""
    input_size: List[int] = field(default_factory=lambda: [512, 1024])
    divisor: int = 32
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class LabelsConfig:
    """标签配置类"""
    dataset: str = "cityscapes"
    road_label: int = 0
    road_label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "cityscapes": 0,
            "bdd100k": 2,
            "mapillary": 20,
        }
    )


@dataclass
class PostProcessingConfig:
    """后处理配置类"""
    confidence_threshold: float = 0.5
    boundary_detection: bool = True
    boundary_method: str = "canny"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    mask_smoothing: bool = True
    smoothing_kernel_size: int = 5


@dataclass
class PerformanceConfig:
    """性能配置类"""
    enable_profiling: bool = True
    device: str = "auto"


@dataclass
class QualityConfig:
    """图像质量检测配置类（用于降级状态判断）"""
    enable_quality_check: bool = True
    brightness_low_threshold: int = 20
    brightness_high_threshold: int = 230
    blur_threshold: float = 30.0
    dark_threshold: int = 30
    bright_threshold: int = 220
    degraded_confidence_threshold: float = 0.1
    unavailable_confidence_threshold: float = 0.05


@dataclass
class SystemConfig:
    """系统配置类"""
    log_level: str = "INFO"
    error_codes: Dict[str, int] = field(
        default_factory=lambda: {
            "success": 0,
            "invalid_input": -1,
            "model_load_failure": -2,
            "inference_error": -3,
            "unknown_error": -99,
        }
    )


@dataclass
class RoadSegmenterConfig:
    """RoadSegmenter模块完整配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # 从YAML文件加载配置
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RoadSegmenterConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    # 从字典加载配置
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RoadSegmenterConfig":
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            labels=LabelsConfig(**config_dict.get("labels", {})),
            post_processing=PostProcessingConfig(
                **config_dict.get("post_processing", {})
            ),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            quality=QualityConfig(**config_dict.get("quality", {})),
            system=SystemConfig(**config_dict.get("system", {})),
        )

    # 获取模型输出中的道路类别Train ID
    def get_road_label(self) -> int:
        return self.labels.road_label_mapping.get(
            self.labels.dataset, self.labels.road_label
        )
