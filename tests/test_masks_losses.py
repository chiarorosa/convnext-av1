import pytest

torch = pytest.importorskip("torch")

from convnext_av1.losses import RDWeightedLoss
from convnext_av1.masks import create_partition_mask


def test_partition_mask_blocks_full_modes_at_frame_edge():
    coords = torch.tensor([[96, 96]])
    frame_size = torch.tensor([[128, 128]])
    block_size = torch.tensor([[64, 64]])
    mask = create_partition_mask(coords, frame_size, block_size)
    assert not bool(mask[0, 0])
    assert not bool(mask[0, 3])
    assert bool(mask[0, 1])
    assert bool(mask[0, 2])


def test_rd_weighted_loss_matches_expected_cost():
    cost = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    loss_fn = RDWeightedLoss(cost, reduction="none")
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    loss = loss_fn(logits, targets)
    assert torch.all(loss < 1e-6)
