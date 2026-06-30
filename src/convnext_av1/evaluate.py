"""Evaluation helpers."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .data import AV1PartitionDataset
from .masks import create_partition_mask
from .metrics import LEVEL_TO_OUTPUT, summarize_predictions
from .model import ConvNeXtAV1
from .train import load_config


def evaluate_from_config(config_path: str | Path, checkpoint_path: str | Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset_path = config["data"].get("test") or config["data"].get("val") or config["data"]["train"]
    dataset = AV1PartitionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 32)), shuffle=False)

    model = ConvNeXtAV1(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_targets, all_preds, all_levels, all_qindices = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["block_y"].to(device), batch["qp"].to(device))
            mask = create_partition_mask(batch["coords"].to(device), batch["frame_size"].to(device), batch["block_size"].to(device))
            levels = batch["level"]
            for level_value, output_name in LEVEL_TO_OUTPUT.items():
                level_mask = levels == level_value
                if not level_mask.any():
                    continue
                logits = outputs[output_name][level_mask.to(device)].masked_fill(~mask[level_mask.to(device)], -1e9)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(batch["partition_mode"][level_mask].numpy().tolist())
                all_levels.extend([level_value] * len(preds))
                qindices = np.rint(batch["qp"][level_mask].numpy().reshape(-1) * 255.0).astype(int)
                all_qindices.extend(qindices.tolist())

    summary = summarize_predictions(all_targets, all_preds, all_levels, all_qindices)
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
