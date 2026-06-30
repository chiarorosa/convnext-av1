"""Conversion from libaom binary logs to validated NumPy datasets."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np

from .binary import SCHEMA_VERSION, RECORD_SIZE, file_sha256, iter_partition_log, write_json


def convert_partition_log(
    bin_path: str | Path,
    output_npz: str | Path,
    manifest_path: str | Path | None = None,
    max_samples: int | None = None,
    export_pickle: str | Path | None = None,
) -> dict[str, Any]:
    bin_path = Path(bin_path)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    samples = list(iter_partition_log(bin_path, max_samples=max_samples))
    if not samples:
        raise ValueError("no samples found in partition log")

    block_y = np.stack([s.y_data.astype(np.float32) / 255.0 for s in samples])[:, None, :, :]
    partition_mode = np.asarray([s.partition_mode for s in samples], dtype=np.int64)
    qindex = np.asarray([s.qindex for s in samples], dtype=np.uint8)
    qindex_norm = qindex.astype(np.float32) / 255.0
    level = np.asarray([s.level for s in samples], dtype=np.int64)
    coords = np.asarray([(s.blk_x, s.blk_y) for s in samples], dtype=np.int64)
    frame_size = np.asarray([(s.frame_width, s.frame_height) for s in samples], dtype=np.int64)
    block_size = np.asarray([(s.blk_w, s.blk_h) for s in samples], dtype=np.int64)
    sequence_id = np.asarray(["unknown"] * len(samples), dtype="U64")

    np.savez_compressed(
        output_npz,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        block_y=block_y,
        partition_mode=partition_mode,
        qindex=qindex,
        qindex_norm=qindex_norm,
        level=level,
        coords=coords,
        frame_size=frame_size,
        block_size=block_size,
        sequence_id=sequence_id,
    )

    if export_pickle is not None:
        export_pickle = Path(export_pickle)
        export_pickle.parent.mkdir(parents=True, exist_ok=True)
        with export_pickle.open("wb") as handle:
            pickle.dump(
                [
                    {
                        "block_y": block_y[i, 0],
                        "partition_mode": int(partition_mode[i]),
                        "qp": float(qindex_norm[i]),
                        "qindex": int(qindex[i]),
                        "level": int(level[i]),
                        "coords": tuple(map(int, coords[i])),
                        "frame_size": tuple(map(int, frame_size[i])),
                        "block_size": tuple(map(int, block_size[i])),
                    }
                    for i in range(len(samples))
                ],
                handle,
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": str(bin_path),
        "input_sha256": file_sha256(bin_path),
        "output": str(output_npz),
        "record_size_bytes": RECORD_SIZE,
        "sample_count": len(samples),
        "level_counts": {str(k): int((level == k).sum()) for k in (0, 1, 2)},
        "qindex_values": sorted(int(v) for v in np.unique(qindex)),
        "partition_mode_counts": {str(k): int((partition_mode == k).sum()) for k in range(10)},
    }
    if manifest_path is None:
        manifest_path = output_npz.with_suffix(".json")
    write_json(manifest_path, manifest)
    return manifest
