"""
文件名: download_checkpoint.py
用途: 下载预训练模型权重
作者: 温涵清
创建日期: 2026-07-16
最后修改日期: 2026-07-17
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT_URLS = {
    "segformer_mit-b2_cityscapes": {
        "hub_id": "smp-hub/segformer-b2-1024x1024-city-160k",
        "filename": "segformer_mit-b2_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b2 pretrained on Cityscapes (19 classes)",
    },
    "segformer_mit-b0_cityscapes": {
        "hub_id": "smp-hub/segformer-b0-1024x1024-city-160k",
        "filename": "segformer_mit-b0_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b0 pretrained on Cityscapes (19 classes)",
    },
    "segformer_mit-b1_cityscapes": {
        "hub_id": "smp-hub/segformer-b1-1024x1024-city-160k",
        "filename": "segformer_mit-b1_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b1 pretrained on Cityscapes (19 classes)",
    },
    "segformer_mit-b3_cityscapes": {
        "hub_id": "smp-hub/segformer-b3-1024x1024-city-160k",
        "filename": "segformer_mit-b3_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b3 pretrained on Cityscapes (19 classes)",
    },
    "segformer_mit-b4_cityscapes": {
        "hub_id": "smp-hub/segformer-b4-1024x1024-city-160k",
        "filename": "segformer_mit-b4_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b4 pretrained on Cityscapes (19 classes)",
    },
    "segformer_mit-b5_cityscapes": {
        "hub_id": "smp-hub/segformer-b5-1024x1024-city-160k",
        "filename": "segformer_mit-b5_cityscapes.pth",
        "num_classes": 19,
        "description": "SegFormer mit-b5 pretrained on Cityscapes (19 classes)",
    },
}


# 下载预训练模型权重
def download_checkpoint(model_name: str, save_dir: str = "checkpoints") -> str:
    config = CHECKPOINT_URLS.get(model_name)
    if config is None:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(CHECKPOINT_URLS.keys())}"
        )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, config["filename"])

    if os.path.exists(save_path):
        print(f"Checkpoint already exists: {save_path}")
        return save_path

    print(f"Downloading checkpoint: {config['description']}")
    print(f"From Hugging Face Hub: {config['hub_id']}")
    return _download_with_smp_from_pretrained(config, save_dir)


# 使用segmentation-models-pytorch下载权重
def _download_with_smp_from_pretrained(config: dict, save_dir: str) -> str:
    import torch
    import segmentation_models_pytorch as smp

    os.makedirs(save_dir, exist_ok=True)

    try:
        model = smp.from_pretrained(config["hub_id"])
        save_path = os.path.join(save_dir, config["filename"])
        torch.save(model.state_dict(), save_path)

        print(f"Checkpoint downloaded successfully to {save_path}")
        print(f"Model architecture: {type(model).__name__}")
        print(f"Number of classes: {config['num_classes']}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        return save_path
    except Exception as e:
        print(f"Failed to download with smp.from_pretrained: {str(e)}")
        print("Trying alternative download method using Hugging Face Hub API...")
        return _download_with_huggingface_hub(config, save_dir)


# 使用Hugging Face Hub下载权重
def _download_with_huggingface_hub(config: dict, save_dir: str) -> str:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    os.makedirs(save_dir, exist_ok=True)

    try:
        checkpoint_path = hf_hub_download(
            repo_id=config["hub_id"],
            filename="model.safetensors",
        )

        checkpoint = load_file(checkpoint_path)
        save_path = os.path.join(save_dir, config["filename"])
        torch.save(checkpoint, save_path)

        print(f"Checkpoint downloaded successfully to {save_path}")
        print(f"Total keys in checkpoint: {len(checkpoint.keys())}")

        return save_path
    except Exception as e:
        print(f"Failed to download with Hugging Face Hub: {str(e)}")
        raise RuntimeError(
            "Failed to download checkpoint. Please try downloading manually from:\n"
            f"https://huggingface.co/{config['hub_id']}"
        )


# 命令行入口函数
def main():
    parser = argparse.ArgumentParser(description="Download segmentation model checkpoint")
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
