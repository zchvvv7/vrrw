"""
文件名: unknown_detector.py
用途: 基于道路分割mask检测未知异常区域
作者: 张楚涵
创建日期: 2026-07-16
最后修改日期: 2026-07-22
"""

from typing import List
from typing import Optional

import cv2
import numpy as np

from src.interface.schemas import UnknownRegion


class UnknownDetector:
    """负责基于道路区域连续性发现未知异常障碍物"""

    def __init__(self, config: dict) -> None:
        self.min_area = config.get("min_area", 600)
        self.max_area = config.get("max_area", 40000)
        self.close_kernel_size = config.get("close_kernel_size", 35)
        self.edge_margin_kernel_size = config.get("edge_margin_kernel_size", 31)
        self.min_score = config.get("min_score", 0.35)
        self.lower_roi_ratio = config.get("lower_roi_ratio", 0.35)

    # 基于道路mask检测未知异常区域
    def predict(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> List[UnknownRegion]:
        height, width = frame.shape[:2]
        road_area = np.where(road_mask > 0, 255, 0).astype(np.uint8)
        expected_road = self._build_expected_road(road_area)
        missing_road = cv2.subtract(expected_road, road_area)
        road_edge = self._build_road_edge(road_area)
        missing_road = cv2.bitwise_and(
            missing_road,
            cv2.bitwise_not(road_edge),
        )
        roi_mask = self._build_roi_mask(height, width)
        candidate_mask = cv2.bitwise_and(missing_road, roi_mask)
        candidate_mask = self._clean_candidate_mask(candidate_mask)
        return self._mask_to_unknown_regions(
            candidate_mask,
            confidence_map,
        )

    # 构造理想连续道路区域
    def _build_expected_road(self, road_area: np.ndarray) -> np.ndarray:
        kernel = np.ones(
            (self.close_kernel_size, self.close_kernel_size),
            dtype=np.uint8,
        )
        expected_road = cv2.morphologyEx(
            road_area,
            cv2.MORPH_CLOSE,
            kernel,
        )
        return expected_road

    # 构造道路边缘抑制区域
    def _build_road_edge(self, road_area: np.ndarray) -> np.ndarray:
        kernel = np.ones(
            (self.edge_margin_kernel_size, self.edge_margin_kernel_size),
            dtype=np.uint8,
        )
        eroded = cv2.erode(road_area, kernel, iterations=1)
        road_edge = cv2.subtract(road_area, eroded)
        return road_edge

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
        confidence_map: Optional[np.ndarray],
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
            if aspect_ratio > 4.5 or aspect_ratio < 0.25:
                continue
            score = min(area / self.max_area, 1.0)
            if confidence_map is not None:
                region_confidence = confidence_map[
                    y:y + box_height,
                    x:x + box_width,
                ]
                if region_confidence.size > 0:
                    confidence_score = 1.0 - float(np.mean(region_confidence))
                    area_score = min(area / self.max_area, 1.0)
                    score = 0.7 * confidence_score + 0.3 * area_score
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