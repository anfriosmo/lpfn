from __future__ import annotations

import torch


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("pred and target must have shape [batch, N, N]")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if pred.shape[-1] != pred.shape[-2]:
        raise ValueError("pred and target matrices must be square")


def frobenius_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared Frobenius error across a batch."""
    _validate_pair(pred, target)
    return (pred - target).abs().square().sum(dim=(-2, -1)).mean()


def phase_insensitive_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean phase-insensitive unitary overlap loss.

    Computes ``1 - |Tr(target^† pred)|^2 / N^2`` for each batch element.
    """
    _validate_pair(pred, target)
    n = pred.shape[-1]
    overlap = torch.diagonal(target.mH @ pred, dim1=-2, dim2=-1).sum(dim=-1)
    return (1.0 - overlap.abs().square() / float(n * n)).mean()
