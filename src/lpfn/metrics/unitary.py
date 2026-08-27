from __future__ import annotations

import torch


def _validate_matrices(U: torch.Tensor) -> None:
    if U.ndim != 3 or U.shape[-1] != U.shape[-2]:
        raise ValueError("U must have shape [batch, N, N]")


def unitarity_defect(U: torch.Tensor) -> torch.Tensor:
    """Operator norm ``||U^†U-I||_op`` for each batch element."""
    _validate_matrices(U)
    n = U.shape[-1]
    eye = torch.eye(n, dtype=U.dtype, device=U.device).expand(U.shape[0], -1, -1)
    defect = U.mH @ U - eye
    return torch.linalg.matrix_norm(defect, ord=2, dim=(-2, -1))


def operator_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Operator norm ``||pred-target||_op`` for each batch element."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    _validate_matrices(pred)
    return torch.linalg.matrix_norm(pred - target, ord=2, dim=(-2, -1))


def phase_insensitive_fidelity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``|Tr(target^†pred)|^2/N^2`` for each batch element."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    _validate_matrices(pred)
    n = pred.shape[-1]
    overlap = torch.diagonal(target.mH @ pred, dim1=-2, dim2=-1).sum(dim=-1)
    return overlap.abs().square() / float(n * n)
