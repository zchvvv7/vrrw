"""
文件名: evaluate.py
用途: 评估RoadSegmenter模块性能，计算mIoU、IoU、Boundary F-score等指标
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import argparse
import os
import sys
import json
from typing import List, Tuple

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.road_segmenter import RoadSegmenter
from src.config.config import RoadSegmenterConfig


# Cityscapes数据集中道路类别的标签ID
CITYSCAPES_ROAD_LABEL = 7


# 计算两个二值掩码的IoU（交并比）
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

# 计算平均IoU（mIoU）
def compute_miou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
                 num_classes: int = 2) -> float:
    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
            iou = compute_iou(pred_cls, gt_cls)
            ious.append(iou)
    return np.mean(ious) if ious else 0.0

# 计算边界F-score
def compute_boundary_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray,
                             boundary_threshold: int = 255,
                             kernel_size: int = 5) -> float:
    pred_boundary = cv2.Canny(pred_mask, 50, 150)
    gt_boundary = cv2.Canny(gt_mask, 50, 150)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gt_boundary_dilated = cv2.dilate(gt_boundary, kernel, iterations=1)

    tp = np.logical_and(pred_boundary > 0, gt_boundary_dilated > 0).sum()
    fp = np.logical_and(pred_boundary > 0, gt_boundary_dilated == 0).sum()
    fn = np.logical_and(pred_boundary == 0, gt_boundary > 0).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_score = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

    return f_score

# 加载统一格式的数据集（images/ 和 masks/ 子目录）
def load_images_masks_format(data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    images = []
    masks = []
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    images.append(img)
                    masks.append(mask)
    return images, masks

# 直接加载Cityscapes原始格式数据集
def load_cityscapes_format(cityscapes_dir: str, split: str = "val") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    images = []
    masks = []
    image_dir = os.path.join(cityscapes_dir, "leftImg8bit", split)
    gt_dir = os.path.join(cityscapes_dir, "gtFine", split)
    if not os.path.exists(image_dir) or not os.path.exists(gt_dir):
        raise ValueError(
            f"Cityscapes directories not found: {image_dir} or {gt_dir}"
        )

    for city in sorted(os.listdir(image_dir)):
        city_image_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        if not os.path.isdir(city_image_dir) or not os.path.isdir(city_gt_dir):
            continue
        for img_file in sorted(os.listdir(city_image_dir)):
            if not img_file.endswith("_leftImg8bit.png"):
                continue
            prefix = img_file.replace("_leftImg8bit.png", "")
            img_path = os.path.join(city_image_dir, img_file)
            label_id_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_labelIds.png")
            polygon_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_polygons.json")
            img = cv2.imread(img_path)
            if img is None:
                continue

            if os.path.exists(label_id_path):
                label_ids = cv2.imread(label_id_path, cv2.IMREAD_GRAYSCALE)
                if label_ids is not None:
                    mask = np.where(
                        label_ids == CITYSCAPES_ROAD_LABEL, 255, 0
                    ).astype(np.uint8)
                else:
                    continue
            elif os.path.exists(polygon_path):
                with open(polygon_path, "r") as f:
                    data = json.load(f)
                height = data["imgHeight"]
                width = data["imgWidth"]
                mask = np.zeros((height, width), dtype=np.uint8)
                for obj in data["objects"]:
                    if obj["label"] == "road":
                        polygon = np.array(
                            obj["polygon"], dtype=np.int32
                        ).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [polygon], 255)
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(
                        mask, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
            else:
                continue

            images.append(img)
            masks.append(mask)

    return images, masks

# 自动检测数据集格式
def detect_dataset_format(data_dir: str) -> str:
    if (os.path.exists(os.path.join(data_dir, "images")) and
            os.path.exists(os.path.join(data_dir, "masks"))):
        return "images_masks"
    elif (os.path.exists(os.path.join(data_dir, "leftImg8bit")) and
            os.path.exists(os.path.join(data_dir, "gtFine"))):
        return "cityscapes"
    else:
        return "unknown"

# 根据格式加载数据集
def load_dataset(data_dir: str, dataset_format: str = None,
                 split: str = "val") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if dataset_format is None:
        dataset_format = detect_dataset_format(data_dir)

    if dataset_format == "images_masks":
        return load_images_masks_format(data_dir)
    elif dataset_format == "cityscapes":
        return load_cityscapes_format(data_dir, split)
    else:
        raise ValueError(
            f"Unknown dataset format. Expected 'images_masks' or 'cityscapes'. "
            f"Data directory should contain 'images/' and 'masks/' subdirs, "
            f"or be a Cityscapes root directory with 'leftImg8bit/' "
            f"and 'gtFine/'."
        )

# 评估RoadSegmenter在给定数据集上的性能
def evaluate_segmenter(segmenter: RoadSegmenter, images: List[np.ndarray],
                       gt_masks: List[np.ndarray]) -> dict:
    pred_masks = []
    ious = []
    boundary_f_scores = []
    inference_times = []

    for img, gt_mask in zip(images, gt_masks):
        result = segmenter.predict(img)
        pred_masks.append(result.mask)
        ious.append(compute_iou(result.mask > 127, gt_mask > 127))
        boundary_f_scores.append(compute_boundary_f_score(result.mask, gt_mask))
        inference_times.append(result.inference_time_ms)

    results = {
        "mean_iou": np.mean(ious),
        "mIoU": compute_miou(pred_masks, gt_masks),
        "mean_boundary_f_score": np.mean(boundary_f_scores),
        "mean_inference_time_ms": np.mean(inference_times),
        "fps": (1000.0 / np.mean(inference_times)
                if np.mean(inference_times) > 0 else 0),
        "num_samples": len(images),
    }

    return results

# 命令行入口函数
def main():
    parser = argparse.ArgumentParser(description="Evaluate RoadSegmenter performance")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/road_segmenter.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to evaluation dataset directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default=None,
        help=("Dataset format: 'images_masks' or 'cityscapes'. "
              "Auto-detected if not specified.")
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split for Cityscapes format: 'train', 'val', or 'test'"
    )
    args = parser.parse_args()

    config = RoadSegmenterConfig.from_yaml(args.config)
    segmenter = RoadSegmenter(config=config)
    if args.checkpoint is not None:
        segmenter.load_checkpoint(args.checkpoint)
    images, gt_masks = load_dataset(args.data_dir, args.dataset_format, args.split)
    print(f"Loaded {len(images)} evaluation samples")
    results = evaluate_segmenter(segmenter, images, gt_masks)

    print("\nEvaluation Results:")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"Mean Boundary F-score: {results['mean_boundary_f_score']:.4f}")
    print(f"Mean Inference Time: {results['mean_inference_time_ms']:.2f} ms")
    print(f"FPS: {results['fps']:.2f}")


if __name__ == "__main__":
    main()