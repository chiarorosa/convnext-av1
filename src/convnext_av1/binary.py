"""Binary contract for libaom partition samples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterator

import numpy as np

SCHEMA_VERSION = 1
MAX_BLOCK_SIZE = 64
HEADER_FORMAT = "<HHHHHHBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
Y_DATA_SIZE = MAX_BLOCK_SIZE * MAX_BLOCK_SIZE
RECORD_SIZE = HEADER_SIZE + Y_DATA_SIZE * np.dtype("<i2").itemsize
PARTITION_CLASSES = 10


@dataclass(frozen=True)
class PartitionSample:
    frame_width: int
    frame_height: int
    blk_x: int
    blk_y: int
    blk_w: int
    blk_h: int
    partition_mode: int
    qindex: int
    level: int
    y_data: np.ndarray

    @property
    def qindex_norm(self) -> float:
        return float(self.qindex) / 255.0

    def validate(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not (0 < self.blk_w <= MAX_BLOCK_SIZE and 0 < self.blk_h <= MAX_BLOCK_SIZE):
            raise ValueError("block dimensions must be in 1..64")
        if self.blk_x < 0 or self.blk_y < 0:
            raise ValueError("block coordinates must be non-negative")
        if self.blk_x >= self.frame_width or self.blk_y >= self.frame_height:
            raise ValueError("block origin is outside the frame")
        if not (0 <= self.partition_mode < PARTITION_CLASSES):
            raise ValueError("partition_mode must be in 0..9")
        if not (0 <= self.qindex <= 255):
            raise ValueError("qindex must be in 0..255")
        if self.level not in (0, 1, 2):
            raise ValueError("level must be one of 0, 1, 2")
        if self.y_data.shape != (MAX_BLOCK_SIZE, MAX_BLOCK_SIZE):
            raise ValueError("y_data must have shape 64x64")
        if self.y_data.min(initial=0) < 0 or self.y_data.max(initial=0) > 255:
            raise ValueError("8-bit y_data must be in 0..255")


def parse_record(record: bytes) -> PartitionSample:
    if len(record) != RECORD_SIZE:
        raise ValueError(f"record has {len(record)} bytes, expected {RECORD_SIZE}")
    fields = struct.unpack(HEADER_FORMAT, record[:HEADER_SIZE])
    y_data = np.frombuffer(record[HEADER_SIZE:], dtype="<i2").reshape(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE).copy()
    sample = PartitionSample(
        frame_width=fields[0],
        frame_height=fields[1],
        blk_x=fields[2],
        blk_y=fields[3],
        blk_w=fields[4],
        blk_h=fields[5],
        partition_mode=fields[6],
        qindex=fields[7],
        level=fields[8],
        y_data=y_data,
    )
    sample.validate()
    return sample


def pack_record(sample: PartitionSample) -> bytes:
    sample.validate()
    header = struct.pack(
        HEADER_FORMAT,
        sample.frame_width,
        sample.frame_height,
        sample.blk_x,
        sample.blk_y,
        sample.blk_w,
        sample.blk_h,
        sample.partition_mode,
        sample.qindex,
        sample.level,
        0,
    )
    return header + sample.y_data.astype("<i2", copy=False).tobytes()


def iter_partition_log(path: str | Path, max_samples: int | None = None) -> Iterator[PartitionSample]:
    path = Path(path)
    size = path.stat().st_size
    if size % RECORD_SIZE != 0:
        raise ValueError(f"{path} size {size} is not a multiple of record size {RECORD_SIZE}")
    with path.open("rb") as handle:
        count = 0
        while True:
            if max_samples is not None and count >= max_samples:
                break
            record = handle.read(RECORD_SIZE)
            if not record:
                break
            yield parse_record(record)
            count += 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
