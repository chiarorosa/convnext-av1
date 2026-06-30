"""Model export helpers."""

from __future__ import annotations

from pathlib import Path

from .model import ConvNeXtAV1
from .train import load_config


def export_checkpoint(config_path: str | Path, checkpoint_path: str | Path, output_path: str | Path, fmt: str = "torchscript") -> str:
    import torch

    config = load_config(config_path)
    model = ConvNeXtAV1(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, 64, 64)
    qp = torch.zeros(1, 1)
    if fmt == "torchscript":
        traced = torch.jit.trace(model, (x, qp), strict=False)
        traced.save(str(output_path))
    elif fmt == "onnx":
        torch.onnx.export(
            model,
            (x, qp),
            str(output_path),
            input_names=["block_y", "qp"],
            output_names=["partition_64x64", "partition_32x32", "partition_16x16"],
            opset_version=17,
        )
    else:
        raise ValueError("fmt must be torchscript or onnx")
    return str(output_path)
