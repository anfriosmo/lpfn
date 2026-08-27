from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .base import ControlModel


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLPControls(ControlModel):
    """Generic MLP control model with output shape ``[batch, K, r]``."""

    def __init__(
        self,
        *,
        input_dim: int,
        depth: int,
        num_generators: int,
        hidden_widths: Sequence[int] = (64, 64),
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if any(int(width) < 1 for width in hidden_widths):
            raise ValueError("all hidden widths must be positive")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}; received {activation!r}"
            )
        if not dtype.is_floating_point:
            raise TypeError("MLP parameters require a floating dtype")

        self.input_dim = int(input_dim)
        self.hidden_widths = tuple(int(w) for w in hidden_widths)
        self.activation_name = activation
        output_dim = self.depth * self.num_generators

        layers: list[nn.Module] = []
        previous = self.input_dim
        activation_cls = _ACTIVATIONS[activation]
        for width in self.hidden_widths:
            layers.append(nn.Linear(previous, width, dtype=dtype))
            layers.append(activation_cls())
            previous = width
        layers.append(nn.Linear(previous, output_dim, dtype=dtype))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"x must have input_dim={self.input_dim}; received {int(x.shape[1])}"
            )
        if x.is_complex() or not x.dtype.is_floating_point:
            raise TypeError("x must be a real floating tensor")

        parameter = next(self.parameters())
        if x.device != parameter.device:
            raise ValueError("x and MLPControls parameters must be on the same device")
        if x.dtype != parameter.dtype:
            raise TypeError(
                f"x dtype {x.dtype} must match MLPControls dtype {parameter.dtype}"
            )

        theta = self.network(x).reshape(
            int(x.shape[0]), self.depth, self.num_generators
        )
        return self.validate_output(theta, batch_size=int(x.shape[0]))
