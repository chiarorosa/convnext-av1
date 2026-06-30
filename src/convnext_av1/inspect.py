"""Dataset visual inspection helpers."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from .data import load_npz_dataset


def inspect_dataset(dataset_path: str | Path, output_dir: str | Path, max_samples: int = 16) -> dict:
    arrays = load_npz_dataset(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_samples, arrays["block_y"].shape[0])
    entries = []
    for index in range(count):
        block = arrays["block_y"][index, 0]
        block_u8 = np.clip(block * 255.0, 0, 255).astype(np.uint8)
        path = output_dir / f"sample_{index:05d}.png"
        Image.fromarray(block_u8, mode="L").save(path)
        entries.append(
            {
                "index": index,
                "path": str(path),
                "partition_mode": int(arrays["partition_mode"][index]),
                "qindex_norm": float(arrays["qindex_norm"][index]),
                "level": int(arrays["level"][index]),
                "coords": arrays["coords"][index].astype(int).tolist(),
                "frame_size": arrays["frame_size"][index].astype(int).tolist(),
                "block_size": arrays["block_size"][index].astype(int).tolist(),
                "min": float(block.min()),
                "max": float(block.max()),
                "mean": float(block.mean()),
            }
        )
    report = {
        "dataset": str(dataset_path),
        "sample_count": int(arrays["block_y"].shape[0]),
        "inspected_count": count,
        "entries": entries,
    }
    (output_dir / "inspect_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
