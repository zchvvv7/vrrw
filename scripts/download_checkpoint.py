"""
文件名: download_checkpoint.py
用途: 下载MMSegmentation预训练模型权重
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-16
"""

import argparse
import os
import urllib.request


# 预训练模型权重下载地址映射
CHECKPOINT_URLS = {
    "segformer_mit-b2_cityscapes": (
        "https://download.openmmlab.com/mmsegmentation/v0.5/"
        "segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/"
        "segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211206_081607-cf065e2e.pth"
    ),
    "segformer_mit-b2_cityscapes_bdd100k": (
        "https://download.openmmlab.com/mmsegmentation/v0.5/"
        "segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/"
        "segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211206_081607-cf065e2e.pth"
    ),
}


# 从指定URL下载预训练模型权重
def download_checkpoint(model_name: str, save_dir: str = "checkpoints") -> str:
    url = CHECKPOINT_URLS.get(model_name)
    if url is None:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(CHECKPOINT_URLS.keys())}"
        )
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(url)
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        print(f"Checkpoint already exists: {save_path}")
        return save_path
    print(f"Downloading checkpoint from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Checkpoint downloaded successfully to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download checkpoint: {str(e)}")
        raise


# 命令行入口函数
def main():
    parser = argparse.ArgumentParser(description="Download MMSegmentation checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default="segformer_mit-b2_cityscapes",
        help=f"Model name. Available: {list(CHECKPOINT_URLS.keys())}"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save the checkpoint"
    )
    args = parser.parse_args()
    download_checkpoint(args.model, args.save_dir)


if __name__ == "__main__":
    main()