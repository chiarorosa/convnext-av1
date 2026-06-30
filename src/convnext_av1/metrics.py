"""Metrics for partition classification experiments."""

from __future__ import annotations

import numpy as np

LEVEL_TO_OUTPUT = {
    0: "partition_64x64",
    1: "partition_32x32",
    2: "partition_16x16",
}


def confusion_matrix(targets, preds, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(np.asarray(targets).reshape(-1), np.asarray(preds).reshape(-1), strict=False):
        matrix[int(target), int(pred)] += 1
    return matrix


def accuracy(targets, preds) -> float:
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if targets.size == 0:
        return 0.0
    return float((targets == preds).mean())


def summarize_predictions(targets, preds, levels, qindices) -> dict:
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    levels = np.asarray(levels)
    qindices = np.asarray(qindices)
    summary = {
        "accuracy": accuracy(targets, preds),
        "sample_count": int(targets.size),
        "by_level": {},
        "by_qindex": {},
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }
    for level in sorted(np.unique(levels).tolist()):
        mask = levels == level
        summary["by_level"][str(int(level))] = {
            "accuracy": accuracy(targets[mask], preds[mask]),
            "sample_count": int(mask.sum()),
        }
    for qindex in sorted(np.unique(qindices).tolist()):
        mask = qindices == qindex
        summary["by_qindex"][str(int(qindex))] = {
            "accuracy": accuracy(targets[mask], preds[mask]),
            "sample_count": int(mask.sum()),
        }
    return summary
