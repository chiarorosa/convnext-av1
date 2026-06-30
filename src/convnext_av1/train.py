"""Training loop for ConvNeXt-AV1."""

from __future__ import annotations

from pathlib import Path
import json
import random
import subprocess
from typing import Any

import numpy as np
import yaml

from .data import AV1PartitionDataset
from .losses import RDWeightedLoss, default_av1_cost_matrix
from .masks import create_partition_mask
from .metrics import LEVEL_TO_OUTPUT
from .model import ConvNeXtAV1


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def batch_loss(model, batch, criterion, device, level_weights: dict[str, float]):
    import torch

    block_y = batch["block_y"].to(device)
    qp = batch["qp"].to(device)
    targets = batch["partition_mode"].to(device)
    levels = batch["level"].to(device)
    coords = batch["coords"].to(device)
    frame_size = batch["frame_size"].to(device)
    block_size = batch["block_size"].to(device)
    outputs = model(block_y, qp)
    mask = create_partition_mask(coords, frame_size, block_size)
    total = torch.zeros((), device=device)
    active = 0
    for level_value, output_name in LEVEL_TO_OUTPUT.items():
        level_mask = levels == level_value
        if not level_mask.any():
            continue
        logits = outputs[output_name][level_mask]
        logits = logits.masked_fill(~mask[level_mask], -1e9)
        total = total + float(level_weights.get(output_name, 1.0)) * criterion(logits, targets[level_mask])
        active += 1
    if active == 0:
        return total
    return total / active


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AV1PartitionDataset(config["data"]["train"])
    val_dataset = AV1PartitionDataset(config["data"].get("val", config["data"]["train"]))
    train_loader = DataLoader(train_dataset, batch_size=int(config["training"].get("batch_size", 32)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["training"].get("batch_size", 32)), shuffle=False)

    model = ConvNeXtAV1(pretrained=bool(config["model"].get("pretrained", False))).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    criterion = RDWeightedLoss(default_av1_cost_matrix().to(device))
    level_weights = config["training"].get("level_weights", {})
    epochs = int(config["training"].get("epochs", 1))
    history = []
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, criterion, device, level_weights)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(batch_loss(model, batch, criterion, device, level_weights).detach().cpu()))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        }
        history.append(row)
        if row["val_loss"] <= best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "history": history,
                    "git_commit": git_commit(),
                },
                checkpoint_dir / "best.pt",
            )

    report = {"config": str(config_path), "history": history, "best_checkpoint": str(checkpoint_dir / "best.pt")}
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
