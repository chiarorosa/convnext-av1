from pathlib import Path

import numpy as np
import pytest

from convnext_av1.binary import PartitionSample, pack_record
from convnext_av1.cli import main


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from convnext_av1.model import ConvNeXtAV1


def test_model_outputs_three_partition_heads():
    model = ConvNeXtAV1(pretrained=False)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, 1, 64, 64), torch.zeros(2, 1))
    assert set(outputs) == {"partition_64x64", "partition_32x32", "partition_16x16"}
    assert outputs["partition_64x64"].shape == (2, 10)


def test_cli_convert_and_inspect(tmp_path: Path):
    y = np.ones((64, 64), dtype=np.int16) * 64
    sample = PartitionSample(128, 128, 0, 0, 64, 64, 0, 64, 0, y)
    bin_path = tmp_path / "partition.bin"
    out = tmp_path / "dataset.npz"
    inspect_dir = tmp_path / "inspect"
    bin_path.write_bytes(pack_record(sample))
    assert main(["convert", str(bin_path), str(out)]) == 0
    assert main(["inspect", str(out), str(inspect_dir), "--max-samples", "1"]) == 0
    assert (inspect_dir / "sample_00000.png").exists()
    assert (inspect_dir / "inspect_report.json").exists()
