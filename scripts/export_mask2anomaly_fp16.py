"""
文件名: export_mask2anomaly_fp16.py
用途: 导出仅包含FP16推理参数的Mask2Anomaly检查点
作者: 张楚涵
创建日期: 2026-08-05
最后修改日期: 2026-08-05
"""

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


HASH_CHUNK_SIZE = 1024 * 1024


# 读取命令行参数
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Mask2Anomaly FP16 inference weights.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Original Mask2Anomaly checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output FP16 inference checkpoint.",
    )
    return parser.parse_args()


# 加载受信任的本地Mask2Anomaly检查点
def load_checkpoint(input_path: Path) -> dict:
    try:
        checkpoint = torch.load(
            str(input_path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(input_path),
            map_location="cpu",
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model weights."
        )
    if not isinstance(checkpoint["model"], dict):
        raise TypeError(
            "Checkpoint model weights must be a dictionary."
        )
    return checkpoint


# 将模型浮点参数转换为FP16
def convert_model_state(model_state: dict) -> dict:
    converted_state = {}
    for name, value in model_state.items():
        converted_state[name] = convert_state_value(value)
    return converted_state


# 转换单个模型状态值并保留非浮点数据类型
def convert_state_value(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    converted_value = value.detach().cpu()
    if torch.is_floating_point(converted_value):
        converted_value = converted_value.to(
            dtype=torch.float16,
        )
    return converted_value.contiguous()


# 计算文件SHA256校验值
def calculate_sha256(file_path: Path) -> str:
    digest = sha256()
    with file_path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# 导出FP16推理检查点并返回SHA256
def export_fp16_checkpoint(
    input_path: Path,
    output_path: Path,
) -> str:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input checkpoint not found: {input_path}"
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Output path cannot overwrite input checkpoint."
        )

    checkpoint = load_checkpoint(input_path)
    fp16_model_state = convert_model_state(
        checkpoint["model"]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {"model": fp16_model_state},
        str(output_path),
    )
    return calculate_sha256(output_path)


# 执行FP16检查点导出
def main() -> None:
    arguments = parse_arguments()
    output_path = Path(arguments.output)
    digest = export_fp16_checkpoint(
        input_path=Path(arguments.input),
        output_path=output_path,
    )
    size_mib = output_path.stat().st_size / 1024 / 1024
    print(f"Output path: {output_path}")
    print(f"Output size: {size_mib:.2f} MiB")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
