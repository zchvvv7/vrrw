"""
文件名: preprocess_cityscapes.py
用途: 将Cityscapes数据集格式转换为统一的images/masks格式
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import cv2


# Cityscapes数据集完整标签映射
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


# 道路类别标签列表
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
                    mask = convert_label_ids_to_road_mask(label_ids)
                else:
                    continue
            elif os.path.exists(polygon_path):
                mask, h, w = parse_cityscapes_polygon(polygon_path)
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(
                        mask, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
            else:
                continue

            output_img_path = os.path.join(output_image_dir, f"{prefix}.png")
            output_mask_path = os.path.join(output_mask_dir, f"{prefix}.png")
            cv2.imwrite(output_img_path, img)
            cv2.imwrite(output_mask_path, mask)
            processed_count += 1

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
        default=["train", "val", "test"],
        help="Splits to process (train, val, test)"
    )
    args = parser.parse_args()

    print(f"Processing Cityscapes dataset from {args.cityscapes_dir}")
    print(f"Output directory: {args.output_dir}")

    for split in args.splits:
        split_output_dir = os.path.join(args.output_dir, split)
        count = process_cityscapes_split(args.cityscapes_dir, split, split_output_dir)
        print(f"Processed {count} samples for {split} split -> {split_output_dir}")

    print("\nPreprocessing completed!")
    print(f"Dataset structure:")
    print(f"  {args.output_dir}/")
    print(f"    train/")
    print(f"      images/  (RGB images)")
    print(f"      masks/   (binary road masks)")
    print(f"    val/")
    print(f"      images/")
    print(f"      masks/")
    print(f"    test/")
    print(f"      images/")
    print(f"      masks/")


if __name__ == "__main__":
    main()