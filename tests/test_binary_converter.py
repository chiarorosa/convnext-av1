from pathlib import Path

import numpy as np

from convnext_av1.binary import HEADER_SIZE, RECORD_SIZE, PartitionSample, pack_record, parse_record
from convnext_av1.converter import convert_partition_log
from convnext_av1.data import load_npz_dataset


def sample(mode=3, level=0):
    y = np.zeros((64, 64), dtype=np.int16)
    y[:16, :16] = 128
    return PartitionSample(
        frame_width=128,
        frame_height=128,
        blk_x=0,
        blk_y=0,
        blk_w=64,
        blk_h=64,
        partition_mode=mode,
        qindex=112,
        level=level,
        y_data=y,
    )


def test_record_contract_round_trip():
    assert HEADER_SIZE == 16
    assert RECORD_SIZE == 8208
    payload = pack_record(sample())
    parsed = parse_record(payload)
    assert parsed.partition_mode == 3
    assert parsed.qindex_norm == 112 / 255
    assert parsed.y_data.shape == (64, 64)


def test_convert_log_to_npz_and_manifest(tmp_path: Path):
    bin_path = tmp_path / "partition.bin"
    bin_path.write_bytes(pack_record(sample(mode=1, level=0)) + pack_record(sample(mode=2, level=1)))
    out = tmp_path / "dataset.npz"
    manifest = tmp_path / "dataset.json"
    result = convert_partition_log(bin_path, out, manifest)
    arrays = load_npz_dataset(out)
    assert result["sample_count"] == 2
    assert arrays["block_y"].shape == (2, 1, 64, 64)
    assert arrays["partition_mode"].tolist() == [1, 2]
    assert arrays["qindex_norm"][0] == np.float32(112 / 255)
    assert manifest.exists()
