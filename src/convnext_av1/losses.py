"""Loss functions for AV1 partition decisions."""

from __future__ import annotations


def default_av1_cost_matrix():
    import torch

    return torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.5, 1.5],
            [2.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2],
            [2.0, 0.8, 0.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2],
            [2.5, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2],
            [2.0, 1.0, 1.0, 1.0, 0.0, 0.8, 0.8, 0.8, 1.2, 1.2],
            [2.0, 1.0, 1.0, 1.0, 0.8, 0.0, 0.8, 0.8, 1.2, 1.2],
            [2.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.0, 0.8, 1.2, 1.2],
            [2.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.0, 1.2, 1.2],
            [2.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2, 0.0, 0.8],
            [2.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )


def RDWeightedLoss(cost_matrix=None, reduction: str = "mean"):
    import torch.nn as nn

    class _Loss(nn.Module):
        def __init__(self, matrix, reduction_mode):
            super().__init__()
            self.register_buffer("cost_matrix", matrix)
            self.reduction = reduction_mode

        def forward(self, logits, targets):
            import torch.nn.functional as F

            probs = F.softmax(logits, dim=1)
            costs = self.cost_matrix.to(logits.device)[targets]
            loss = (probs * costs).sum(dim=1)
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "none":
                return loss
            raise ValueError(f"unknown reduction: {self.reduction}")

    return _Loss(cost_matrix if cost_matrix is not None else default_av1_cost_matrix(), reduction)
