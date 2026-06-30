"""Dataset and split utilities."""

from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "block_y",
    "partition_mode",
    "qindex_norm",
    "level",
    "coords",
    "frame_size",
    "block_size",
}


def load_npz_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    missing = REQUIRED_ARRAYS.difference(data.files)
    if missing:
        raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
    arrays = {name: data[name] for name in data.files}
    n = arrays["block_y"].shape[0]
    for key in REQUIRED_ARRAYS:
        if arrays[key].shape[0] != n:
            raise ValueError(f"array {key} length does not match block_y")
    return arrays


def create_sequence_split(
    dataset_path: str | Path,
    output_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 1337,
) -> dict[str, Any]:
    arrays = load_npz_dataset(dataset_path)
    sequence_id = arrays.get("sequence_id")
    if sequence_id is None:
        raise ValueError("dataset does not include sequence_id; cannot split safely by sequence")
    sequences = sorted({str(value) for value in sequence_id.tolist()})
    if sequences == ["unknown"]:
        raise ValueError("sequence_id is unknown for all samples; provide real sequence metadata before splitting")

    rng = random.Random(seed)
    rng.shuffle(sequences)
    n_train = max(1, int(round(len(sequences) * train_ratio)))
    n_val = max(1, int(round(len(sequences) * val_ratio))) if len(sequences) > 2 else 0
    train_seq = set(sequences[:n_train])
    val_seq = set(sequences[n_train : n_train + n_val])
    test_seq = set(sequences[n_train + n_val :])
    if not test_seq and len(sequences) > 1:
        test_seq.add(sorted(train_seq).pop())

    splits = {
        "train": [i for i, seq in enumerate(sequence_id.tolist()) if str(seq) in train_seq],
        "val": [i for i, seq in enumerate(sequence_id.tolist()) if str(seq) in val_seq],
        "test": [i for i, seq in enumerate(sequence_id.tolist()) if str(seq) in test_seq],
    }
    payload = {
        "dataset": str(dataset_path),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "split_by": "sequence_id",
        "sequences": {
            "train": sorted(train_seq),
            "val": sorted(val_seq),
            "test": sorted(test_seq),
        },
        "indices": splits,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


class AV1PartitionDataset:
    """Torch-compatible dataset for converted partition samples."""

    def __init__(self, path: str | Path, indices: list[int] | None = None):
        self.arrays = load_npz_dataset(path)
        self.indices = np.asarray(indices if indices is not None else np.arange(self.arrays["block_y"].shape[0]))

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        import torch

        real_idx = int(self.indices[idx])
        return {
            "block_y": torch.as_tensor(self.arrays["block_y"][real_idx], dtype=torch.float32),
            "partition_mode": torch.as_tensor(self.arrays["partition_mode"][real_idx], dtype=torch.long),
            "qp": torch.as_tensor([self.arrays["qindex_norm"][real_idx]], dtype=torch.float32),
            "level": torch.as_tensor(self.arrays["level"][real_idx], dtype=torch.long),
            "coords": torch.as_tensor(self.arrays["coords"][real_idx], dtype=torch.long),
            "frame_size": torch.as_tensor(self.arrays["frame_size"][real_idx], dtype=torch.long),
            "block_size": torch.as_tensor(self.arrays["block_size"][real_idx], dtype=torch.long),
        }
