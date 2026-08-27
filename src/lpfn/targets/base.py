from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoxDomain:
    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) < 1 or len(self.low) != len(self.high):
            raise ValueError("low and high must have the same positive dimension")
        if any(a >= b for a, b in zip(self.low, self.high)):
            raise ValueError("every domain coordinate must satisfy low < high")

    @property
    def input_dim(self) -> int:
        return len(self.low)


class TargetFamily(ABC):
    """Exact/synthetic matrix-valued target family used by benchmarks."""

    def __init__(self, *, name: str, domain: BoxDomain) -> None:
        self.name = str(name)
        self.domain = domain

    @property
    def input_dim(self) -> int:
        return self.domain.input_dim

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
