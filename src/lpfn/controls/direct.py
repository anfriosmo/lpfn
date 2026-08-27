from __future__ import annotations

from collections.abc import Callable

import torch

from .base import ControlModel

ControlFunction = Callable[[torch.Tensor], torch.Tensor]


class DirectControls(ControlModel):
    """Non-trainable controls supplied explicitly or by a known function.

    This class is intentionally simple: it lets us validate the geometric map
    theta -> product exp(theta B) without introducing optimization.
    """

    def __init__(
        self,
        controls: torch.Tensor | ControlFunction,
        *,
        depth: int,
        num_generators: int,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if not callable(controls) and not isinstance(controls, torch.Tensor):
            raise TypeError("controls must be a torch.Tensor or callable")
        self._control_fn = controls if callable(controls) else None
        if isinstance(controls, torch.Tensor):
            self.register_buffer("_controls", controls)
        else:
            self.register_buffer("_controls", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        batch_size = int(x.shape[0])

        if self._control_fn is not None:
            theta = self._control_fn(x)
            if not isinstance(theta, torch.Tensor):
                raise TypeError("control callable must return a torch.Tensor")
        else:
            theta = self._controls
            assert theta is not None
            if theta.ndim == 2:
                theta = theta.unsqueeze(0).expand(batch_size, -1, -1)
            elif theta.ndim == 3 and theta.shape[0] == 1:
                theta = theta.expand(batch_size, -1, -1)
            elif theta.ndim == 3 and theta.shape[0] == batch_size:
                pass
            else:
                raise ValueError(
                    "direct control tensor must have shape [K,r], [1,K,r], "
                    "or [batch,K,r]"
                )

        return self.validate_output(theta, batch_size=batch_size)
