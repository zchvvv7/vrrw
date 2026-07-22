"""
文件名: unknown_detector.py
用途: 基于深度估计和道路分割结果检测未知异常区域
作者:
创建日期: 2026-07-16
最后修改日期: 2026-07-22
"""

from typing import List
from typing import Optional

import cv2
import numpy as np

from src.interface.schemas import UnknownRegion


class UnknownDetector:
    """负责发现可行驶区域内的未知异常风险"""

    def __init__(self, config: dict) -> None:
        self.method = config.get("method", "depth_baseline")
        self.min_area = config.get("min_area", 600)
        self.max_area = config.get("max_area", 40000)
        self.min_score = config.get("min_score", 0.35)
        self.lower_roi_ratio = config.get("lower_roi_ratio", 0.35)
        self.depth_gradient_threshold = config.get(
            "depth_gradient_threshold",
            0.18,
        )
        self.depth_close_percentile = config.get(
            "depth_close_percentile",
            70,
        )
        self.depth_model = None
        self.depth_available = False
        if self.method == "depth_baseline":
            self._init_depth_model(config)

    # 初始化Depth Anything V2模型
    def _init_depth_model(self, config: dict) -> None:
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
            encoder = config.get("encoder", "vits")
            checkpoint_path = config.get("checkpoint_path")

            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }

            if encoder not in model_configs:
                raise ValueError(f"Unsupported encoder: {encoder}")
            if checkpoint_path is None:
                raise ValueError("Depth checkpoint_path is not configured.")
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            self.depth_model = DepthAnythingV2(**model_configs[encoder])
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.depth_model.load_state_dict(checkpoint)
            self.depth_model = self.depth_model.to(self.device).eval()
            self.depth_available = True

        except Exception:
            self.depth_model = None
            self.depth_available = False

    # 检测未知异常区域
    def predict(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> List[UnknownRegion]:
        if self.method == "depth_baseline" and self.depth_available:
            return self._predict_by_depth(frame, road_mask)
        return self._predict_by_confidence(
            frame,
            road_mask,
            confidence_map,
        )

    # 基于Depth Anything V2检测道路内突起区域
    def _predict_by_depth(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
    ) -> List[UnknownRegion]:
        depth_map = self._estimate_depth(frame)
        depth_map = self._normalize_depth(depth_map)
        height, width = frame.shape[:2]
        road_area = np.where(road_mask > 0, 255, 0).astype(np.uint8)
        roi_mask = self._build_roi_mask(height, width)
        valid_area = cv2.bitwise_and(road_area, roi_mask)
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        depth_gradient = cv2.magnitude(grad_x, grad_y)

        gradient_candidate = np.where(
            depth_gradient > self.depth_gradient_threshold,
            255,
            0,
        ).astype(np.uint8)
        road_depth_values = depth_map[valid_area > 0]
        if road_depth_values.size == 0:
            return []
        close_threshold = np.percentile(
            road_depth_values,
            self.depth_close_percentile,
        )
        close_candidate = np.where(
            depth_map > close_threshold,
            255,
            0,
        ).astype(np.uint8)
        candidate_mask = cv2.bitwise_and(
            gradient_candidate,
            close_candidate,
        )
        candidate_mask = cv2.bitwise_and(candidate_mask, valid_area)
        candidate_mask = self._clean_candidate_mask(candidate_mask)

        return self._mask_to_unknown_regions(
            candidate_mask,
            score_map=depth_gradient,
        )

    # 调用Depth Anything V2生成深度图
    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        depth_map = self.depth_model.infer_image(frame)
        return depth_map.astype(np.float32)

    # 将深度图归一化到0到1
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        min_value = float(np.min(depth_map))
        max_value = float(np.max(depth_map))
        if max_value - min_value < 1e-6:
            return np.zeros_like(depth_map, dtype=np.float32)
        return (depth_map - min_value) / (max_value - min_value)

    # 基于道路分割置信度的备用检测方法
    def _predict_by_confidence(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
        confidence_map: Optional[np.ndarray],
    ) -> List[UnknownRegion]:
        if confidence_map is None:
            return []
        height, width = frame.shape[:2]
        road_area = np.where(road_mask > 0, 255, 0).astype(np.uint8)
        roi_mask = self._build_roi_mask(height, width)
        valid_area = cv2.bitwise_and(road_area, roi_mask)
        low_confidence = np.where(
            confidence_map < 0.35,
            255,
            0,
        ).astype(np.uint8)
        candidate_mask = cv2.bitwise_and(low_confidence, valid_area)
        candidate_mask = self._clean_candidate_mask(candidate_mask)
        score_map = 1.0 - confidence_map
        return self._mask_to_unknown_regions(
            candidate_mask,
            score_map=score_map,
        )

    # 构造画面下方ROI
    def _build_roi_mask(self, height: int, width: int) -> np.ndarray:
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[int(height * self.lower_roi_ratio):, :] = 255
        return roi_mask

    # 清理候选区域
    def _clean_candidate_mask(self, candidate_mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), dtype=np.uint8)
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            kernel,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
        return candidate_mask

    # 将候选mask转换成UnknownRegion
    def _mask_to_unknown_regions(
        self,
        candidate_mask: np.ndarray,
        score_map: np.ndarray,
    ) -> List[UnknownRegion]:
        contours, _ = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        unknown_regions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 15 or box_height < 15:
                continue
            aspect_ratio = box_width / max(box_height, 1)
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue
            region_score = score_map[
                y:y + box_height,
                x:x + box_width,
            ]
            score = float(np.mean(region_score))
            if score < self.min_score:
                continue
            unknown_regions.append(
                UnknownRegion(
                    bbox=(x, y, x + box_width, y + box_height),
                    score=score,
                    distance=None,
                )
            )

        return unknown_regions