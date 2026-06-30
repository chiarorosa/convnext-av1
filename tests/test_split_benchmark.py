import csv
from pathlib import Path

import numpy as np
import pytest

from convnext_av1.benchmark import consolidate_benchmark
from convnext_av1.data import create_sequence_split


def make_dataset(path: Path):
    n = 6
    np.savez_compressed(
        path,
        block_y=np.zeros((n, 1, 64, 64), dtype=np.float32),
        partition_mode=np.zeros(n, dtype=np.int64),
        qindex=np.zeros(n, dtype=np.uint8),
        qindex_norm=np.zeros(n, dtype=np.float32),
        level=np.zeros(n, dtype=np.int64),
        coords=np.zeros((n, 2), dtype=np.int64),
        frame_size=np.ones((n, 2), dtype=np.int64) * 128,
        block_size=np.ones((n, 2), dtype=np.int64) * 64,
        sequence_id=np.asarray(["a", "a", "b", "b", "c", "c"]),
    )


def test_split_rejects_unknown_sequence_ids(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    n = 2
    np.savez_compressed(
        path,
        block_y=np.zeros((n, 1, 64, 64), dtype=np.float32),
        partition_mode=np.zeros(n, dtype=np.int64),
        qindex_norm=np.zeros(n, dtype=np.float32),
        level=np.zeros(n, dtype=np.int64),
        coords=np.zeros((n, 2), dtype=np.int64),
        frame_size=np.ones((n, 2), dtype=np.int64),
        block_size=np.ones((n, 2), dtype=np.int64),
        sequence_id=np.asarray(["unknown", "unknown"]),
    )
    with pytest.raises(ValueError):
        create_sequence_split(path, tmp_path / "split.json")


def test_sequence_split_and_benchmark(tmp_path: Path):
    dataset = tmp_path / "dataset.npz"
    make_dataset(dataset)
    split = create_sequence_split(dataset, tmp_path / "split.json", seed=1)
    assert split["split_by"] == "sequence_id"
    assert split["indices"]["train"]

    csv_path = tmp_path / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bd_rate", "delta_psnr", "time_reduction"])
        writer.writeheader()
        writer.writerow({"bd_rate": "0.5", "delta_psnr": "-0.03", "time_reduction": "25"})
    report = consolidate_benchmark(csv_path, tmp_path / "benchmark.json")
    assert report["bd_rate_mean"] == 0.5
