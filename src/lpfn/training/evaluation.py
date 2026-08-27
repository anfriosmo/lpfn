from __future__ import annotations

from time import perf_counter

import torch
from torch import nn

from lpfn.losses import frobenius_loss, phase_insensitive_loss
from lpfn.metrics import operator_error, phase_insensitive_fidelity, unitarity_defect


def evaluate_unitary_model(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    start = perf_counter()
    with torch.no_grad():
        pred = model(x)
    runtime = perf_counter() - start
    return {
        "frobenius_loss": float(frobenius_loss(pred, target)),
        "phase_loss": float(phase_insensitive_loss(pred, target)),
        "mean_operator_error": float(operator_error(pred, target).mean()),
        "max_operator_error": float(operator_error(pred, target).max()),
        "mean_phase_fidelity": float(phase_insensitive_fidelity(pred, target).mean()),
        "max_unitarity_defect": float(unitarity_defect(pred).max()),
        "inference_seconds": runtime,
    }
