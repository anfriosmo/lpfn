from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .base import ControlModel
from .spline import _bspline_basis_1d, _open_uniform_knots


class KANControls(ControlModel):
    """Composable Kolmogorov--Arnold control model with spline edge functions.

    A KAN layer maps ``R^{n_in} -> R^{n_out}`` by placing a trainable
    univariate function on every edge and summing edge values at each output
    node.  Here every edge function is represented in a fixed open-uniform
    B-spline basis.

    To make several KAN layers composable on a fixed spline domain without an
    adaptive-grid algorithm, hidden layers use bounded effective spline
    coefficients and average incoming edge contributions.  Because the
    B-spline basis is a partition of unity, hidden node values remain in
    ``[-1,1]``.  The final KAN layer is unconstrained and returns the LPFN
    scalar controls with shape ``[batch, K, r]``.

    ``hidden_widths=()`` gives a single classical KAN layer.  Non-empty hidden
    widths give a genuinely compositional KAN while preserving inspectable
    edge functions.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        depth: int,
        num_generators: int,
        hidden_widths: Sequence[int] = (2,),
        num_basis_per_edge: int = 4,
        degree: int = 3,
        init_scale: float = 0.02,
        hidden_coefficient_scale: float = 1.0,
        validate_domain: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if any(int(w) < 1 for w in hidden_widths):
            raise ValueError("all KAN hidden widths must be positive")
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if num_basis_per_edge < degree + 1:
            raise ValueError("num_basis_per_edge must be at least degree + 1")
        if hidden_coefficient_scale <= 0:
            raise ValueError("hidden_coefficient_scale must be positive")
        if not dtype.is_floating_point:
            raise TypeError("KAN parameters require a floating dtype")

        self.input_dim = int(input_dim)
        self.hidden_widths = tuple(int(w) for w in hidden_widths)
        self.num_basis_per_edge = int(num_basis_per_edge)
        self.degree = int(degree)
        self.hidden_coefficient_scale = float(hidden_coefficient_scale)
        self.validate_domain = bool(validate_domain)
        self.output_dim = self.depth * self.num_generators

        self.register_buffer(
            "knots",
            _open_uniform_knots(self.num_basis_per_edge, self.degree, dtype=dtype),
            persistent=True,
        )

        widths = (self.input_dim, *self.hidden_widths, self.output_dim)
        params: list[nn.Parameter] = []
        for n_in, n_out in zip(widths[:-1], widths[1:]):
            coeff = torch.zeros(n_in, n_out, self.num_basis_per_edge, dtype=dtype)
            if init_scale > 0:
                coeff.normal_(mean=0.0, std=float(init_scale))
            params.append(nn.Parameter(coeff))
        self.coefficients = nn.ParameterList(params)

    @property
    def num_layers(self) -> int:
        return len(self.coefficients)

    @property
    def layer_widths(self) -> tuple[int, ...]:
        return (self.input_dim, *self.hidden_widths, self.output_dim)

    def effective_coefficients(self, layer_index: int) -> torch.Tensor:
        """Return the coefficients that define edge functions in a layer.

        Hidden-layer coefficients are bounded so that intermediate KAN nodes
        stay in the spline domain.  Final-layer coefficients are unconstrained.
        """
        if not (0 <= int(layer_index) < self.num_layers):
            raise IndexError("layer_index out of range")
        raw = self.coefficients[int(layer_index)]
        if int(layer_index) < self.num_layers - 1:
            return self.hidden_coefficient_scale * torch.tanh(raw)
        return raw

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"x must have input_dim={self.input_dim}; received {int(x.shape[1])}"
            )
        if x.is_complex() or not x.dtype.is_floating_point:
            raise TypeError("x must be a real floating tensor")
        parameter = self.coefficients[0]
        if x.device != parameter.device:
            raise ValueError("x and KANControls parameters must be on the same device")
        if x.dtype != parameter.dtype:
            raise TypeError(
                f"x dtype {x.dtype} must match KANControls dtype {parameter.dtype}"
            )
        if self.validate_domain:
            tol = 1e-12 if x.dtype == torch.float64 else 1e-6
            if torch.any(x < -1.0 - tol) or torch.any(x > 1.0 + tol):
                raise ValueError("KANControls expects inputs in [-1, 1]")

    def _basis_for_nodes(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch, n_nodes] -> basis: [batch, n_nodes, n_basis]
        batch, n_nodes = values.shape
        flat = values.reshape(-1)
        B = _bspline_basis_1d(
            flat, self.knots, self.degree, self.num_basis_per_edge
        )
        return B.reshape(batch, n_nodes, self.num_basis_per_edge)

    def forward_with_edges(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return controls together with per-layer edge contributions.

        Each edge tensor has shape ``[batch, n_in, n_out]`` and is useful for
        interpretability/pruning experiments without changing forward semantics.
        """
        self._validate_input(x)
        values = x
        edge_history: list[torch.Tensor] = []

        for layer_index in range(self.num_layers):
            basis = self._basis_for_nodes(values)
            coeff = self.effective_coefficients(layer_index)
            # edge[b, i, o] = phi_{io}(values[b,i])
            edge = torch.einsum("bin,ion->bio", basis, coeff)
            edge_history.append(edge)
            if layer_index < self.num_layers - 1:
                # Mean is a fixed rescaling of the KAN summation and, together
                # with bounded edge functions, keeps hidden nodes in [-1,1].
                values = edge.mean(dim=1)
            else:
                values = edge.sum(dim=1)

        theta = values.reshape(int(x.shape[0]), self.depth, self.num_generators)
        theta = self.validate_output(theta, batch_size=int(x.shape[0]))
        return theta, tuple(edge_history)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta, _ = self.forward_with_edges(x)
        return theta
