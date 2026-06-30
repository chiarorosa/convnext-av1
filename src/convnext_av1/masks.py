"""AV1 partition validity masks."""

from __future__ import annotations


def create_partition_mask(coords, frame_size, block_size, num_classes: int = 10):
    import torch

    if num_classes != 10:
        raise ValueError("AV1 partition mask currently expects 10 classes")
    coords = coords.long()
    frame_size = frame_size.long()
    block_size = block_size.long()
    device = coords.device
    batch = coords.shape[0]
    mask = torch.ones((batch, num_classes), dtype=torch.bool, device=device)

    x, y = coords[:, 0], coords[:, 1]
    frame_w, frame_h = frame_size[:, 0], frame_size[:, 1]
    block_w, block_h = block_size[:, 0], block_size[:, 1]

    fits_full = (x + block_w <= frame_w) & (y + block_h <= frame_h)
    fits_half_h = y + (block_h + 1) // 2 <= frame_h
    fits_half_w = x + (block_w + 1) // 2 <= frame_w
    fits_quarter_h = y + (block_h + 3) // 4 <= frame_h
    fits_quarter_w = x + (block_w + 3) // 4 <= frame_w

    mask[~fits_full, 0] = False  # NONE
    mask[~fits_half_h, 1] = False  # HORZ
    mask[~fits_half_w, 2] = False  # VERT
    mask[~fits_full, 3] = False  # SPLIT
    mask[~fits_full, 4] = False  # HORZ_A
    mask[~fits_full, 5] = False  # HORZ_B
    mask[~fits_full, 6] = False  # VERT_A
    mask[~fits_full, 7] = False  # VERT_B
    mask[~fits_quarter_h, 8] = False  # HORZ_4
    mask[~fits_quarter_w, 9] = False  # VERT_4
    return mask
