from __future__ import annotations

import torch
from torch import nn

from lpfn.controls import ControlModel
from lpfn.engines import ExecutionEngine, TorchMatrixEngine
from lpfn.generators import GeneratorSet


class LieProductNetwork(nn.Module):
    """Mother LPFN model: scalar controls followed by fixed Lie geometry.

    The module composes

        x -> ControlModel(x) = theta -> ExecutionEngine(theta) = U(x).

    The control model is trainable; the generator set and execution engine fix
    the geometric semantics. Changing the control model must not change the
    Lie-product implementation.
    """

    def __init__(
        self,
        *,
        generators: GeneratorSet,
        controls: ControlModel,
        engine: ExecutionEngine | None = None,
        base_point: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if controls.num_generators != generators.num_generators:
            raise ValueError(
                "controls.num_generators must equal generators.num_generators"
            )
        self.generators = generators
        self.controls = controls
        self.engine = engine if engine is not None else TorchMatrixEngine()

        if base_point is not None and not isinstance(base_point, torch.Tensor):
            raise TypeError("base_point must be a torch.Tensor or None")
        self.register_buffer("_base_point", base_point)

    @property
    def depth(self) -> int:
        return self.controls.depth

    @property
    def num_generators(self) -> int:
        return self.generators.num_generators

    @property
    def base_point(self) -> torch.Tensor | None:
        return self._base_point

    def control_values(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        theta = self.controls(x)
        return self.controls.validate_output(theta, batch_size=int(x.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.control_values(x)
        return self.engine.execute(
            theta,
            self.generators,
            base_point=self._base_point,
        )

    def trainable_parameter_count(self) -> int:
        return self.controls.trainable_parameter_count()
