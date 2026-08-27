from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ControlModel(nn.Module, ABC):
    """Abstract scalar-control model x -> theta with shape [batch, K, r]."""

    def __init__(self, *, depth: int, num_generators: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if num_generators < 1:
            raise ValueError("num_generators must be at least 1")
        self.depth = int(depth)
        self.num_generators = int(num_generators)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def validate_output(self, theta: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        expected = (batch_size, self.depth, self.num_generators)
        if tuple(theta.shape) != expected:
            raise ValueError(
                f"ControlModel must return shape {expected}; received {tuple(theta.shape)}"
            )
        if theta.is_complex():
            raise TypeError("LPFN scalar controls theta must be real-valued")
        if not theta.dtype.is_floating_point:
            raise TypeError("LPFN scalar controls theta must use a floating dtype")
        return theta

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
