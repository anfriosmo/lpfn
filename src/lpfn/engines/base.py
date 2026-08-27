from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lpfn.generators import GeneratorSet


class ExecutionEngine(ABC):
    """Maps controls to the geometric Lie-product output."""

    @abstractmethod
    def execute(
        self,
        theta: torch.Tensor,
        generators: GeneratorSet,
        *,
        base_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
