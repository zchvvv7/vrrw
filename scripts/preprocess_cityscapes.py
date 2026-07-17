"""
文件名: preprocess_cityscapes.py
用途: 将Cityscapes数据集格式转换为统一的images/masks格式
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-17
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np


CITYSCAPES_LABELS: Dict[int, str] = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
    -1: "license plate",
}


ROAD_LABELS = [7]


# 将Cityscapes标签ID图像转换为二值道路掩码
def convert_label_ids_to_road_mask(label_ids: np.ndarray) -> np.ndarray:
    mask = np.zeros(label_ids.shape, dtype=np.uint8)
    for road_label in ROAD_LABELS:
        mask[label_ids == road_label] = 255
    return mask


# 解析Cityscapes多边形标注文件
def parse_cityscapes_polygon(json_path: str) -> Tuple[np.ndarray, int, int]:
    with open(json_path, "r") as f:
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

    return mask, height, width


# 处理Cityscapes数据集中的一个划分（train/val/test）
def process_cityscapes_split(cityscapes_dir: str, split: str,
                             output_dir: str) -> int:
    image_dir = os.path.join(cityscapes_dir, "leftImg8bit", split)
    gt_dir = os.path.join(cityscapes_dir, "gtFine", split)

    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    if not os.path.exists(image_dir):
        print(f"  Warning: Image directory not found: {image_dir}")
        return 0

    cities = sorted(os.listdir(image_dir))
    total_cities = len(cities)

    for city_idx, city in enumerate(cities):
        city_image_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)

        if not os.path.isdir(city_image_dir):
            continue

        if not os.path.isdir(city_gt_dir):
            print(f"  Warning: GT directory not found for city '{city}', skipping...")
            skipped_count += len(os.listdir(city_image_dir))
            continue

        img_files = sorted([f for f in os.listdir(city_image_dir)
                           if f.endswith("_leftImg8bit.png")])

        for img_file in img_files:
            prefix = img_file.replace("_leftImg8bit.png", "")
            img_path = os.path.join(city_image_dir, img_file)

            label_id_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_labelIds.png")
            polygon_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_polygons.json")

            try:
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue

                mask = None
                if os.path.exists(label_id_path):
                    label_ids = cv2.imread(label_id_path, cv2.IMREAD_GRAYSCALE)
                    if label_ids is not None:
                        mask = convert_label_ids_to_road_mask(label_ids)
                elif os.path.exists(polygon_path):
                    mask, h, w = parse_cityscapes_polygon(polygon_path)
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(
                            mask, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                if mask is None:
                    skipped_count += 1
                    continue

                output_img_path = os.path.join(output_image_dir, f"{prefix}.png")
                output_mask_path = os.path.join(output_mask_dir, f"{prefix}.png")

                success = cv2.imwrite(output_img_path, img)
                if not success:
                    raise RuntimeError(f"Failed to write image: {output_img_path}")

                success = cv2.imwrite(output_mask_path, mask)
                if not success:
                    raise RuntimeError(f"Failed to write mask: {output_mask_path}")

                processed_count += 1

            except Exception as e:
                error_count += 1
                print(f"  Error processing {prefix}: {str(e)}")

        progress = ((city_idx + 1) / total_cities) * 100
        print(f"  Progress: {city_idx + 1}/{total_cities} cities ({progress:.1f}%) - "
              f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}",
              end="\r")

    print(f"\n  Final: Processed {processed_count}, Skipped {skipped_count}, Errors {error_count}")
    return processed_count


# 命令行入口函数
def main():
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes dataset to images/masks format"
    )
    parser.add_argument(
        "--cityscapes_dir",
        type=str,
        required=True,
        help="Path to Cityscapes root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/cityscapes",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Splits to process (train, val, test). Note: test split typically has no labels."
    )
    args = parser.parse_args()

    print(f"Processing Cityscapes dataset from {args.cityscapes_dir}")
    print(f"Output directory: {args.output_dir}")

    total_processed = 0
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        split_output_dir = os.path.join(args.output_dir, split)
        count = process_cityscapes_split(args.cityscapes_dir, split, split_output_dir)
        print(f"Processed {count} samples for {split} split -> {split_output_dir}")
        total_processed += count

    print(f"\nPreprocessing completed!")
    print(f"Total samples processed: {total_processed}")
    print(f"Dataset structure:")
    print(f"  {args.output_dir}/")
    for split in args.splits:
        print(f"    {split}/")
        print(f"      images/  (RGB images)")
        print(f"      masks/   (binary road masks)")


if __name__ == "__main__":
    main()
