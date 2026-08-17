"""
文件名: test_export_mask2anomaly_fp16.py
用途: 测试Mask2Anomaly FP16推理权重导出
作者: 张楚涵
创建日期: 2026-08-05
最后修改日期: 2026-08-05
"""

from pathlib import Path

import torch

from scripts.export_mask2anomaly_fp16 import (
    calculate_sha256,
)
from scripts.export_mask2anomaly_fp16 import (
    export_fp16_checkpoint,
)
from scripts.export_mask2anomaly_fp16 import load_checkpoint


# 测试导出时删除训练状态并转换浮点张量
def test_export_fp16_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "training.pth"
    output_path = tmp_path / "inference_fp16.pth"
    torch.save(
        {
            "model": {
                "weight": torch.tensor(
                    [1.0, 2.0],
                    dtype=torch.float32,
                ),
                "counter": torch.tensor(
                    [3],
                    dtype=torch.int64,
                ),
            },
            "trainer": {
                "optimizer": "unused",
            },
            "iteration": 100,
        },
        input_path,
    )

    digest = export_fp16_checkpoint(
        input_path=input_path,
        output_path=output_path,
    )
    exported = load_checkpoint(output_path)

    assert set(exported) == {"model"}
    assert exported["model"]["weight"].dtype == torch.float16
    assert exported["model"]["counter"].dtype == torch.int64
    assert digest == calculate_sha256(output_path)


# 测试导出过程禁止覆盖原始权重
def test_export_rejects_source_overwrite(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "model": {
                "weight": torch.ones(1),
            }
        },
        checkpoint_path,
    )

    try:
        export_fp16_checkpoint(
            input_path=checkpoint_path,
            output_path=checkpoint_path,
        )
    except ValueError as error:
        assert "cannot overwrite" in str(error)
    else:
        raise AssertionError("Source overwrite must raise ValueError.")
