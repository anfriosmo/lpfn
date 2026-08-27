from __future__ import annotations

from itertools import product

import torch
from torch import nn

from .base import ControlModel


def _open_uniform_knots(num_basis: int, degree: int, *, dtype: torch.dtype) -> torch.Tensor:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if num_basis < degree + 1:
        raise ValueError("num_basis_per_dim must be at least degree + 1")
    n_internal = num_basis - degree - 1
    if n_internal > 0:
        internal = torch.linspace(-1.0, 1.0, n_internal + 2, dtype=dtype)[1:-1]
    else:
        internal = torch.empty(0, dtype=dtype)
    return torch.cat((
        torch.full((degree + 1,), -1.0, dtype=dtype),
        internal,
        torch.full((degree + 1,), 1.0, dtype=dtype),
    ))


def _bspline_basis_1d(x: torch.Tensor, knots: torch.Tensor, degree: int, num_basis: int) -> torch.Tensor:
    """Cox-de Boor basis evaluation for a fixed open knot vector."""
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    t = knots.to(device=x.device, dtype=x.dtype)
    n0 = int(t.numel() - 1)
    # Half-open intervals. Endpoint x=1 is repaired after the recurrence.
    B = ((x[:, None] >= t[:-1]) & (x[:, None] < t[1:])).to(dtype=x.dtype)
    for p in range(1, degree + 1):
        n_new = n0 - p
        cols: list[torch.Tensor] = []
        for i in range(n_new):
            left_denom = t[i + p] - t[i]
            right_denom = t[i + p + 1] - t[i + 1]
            left = torch.zeros_like(x)
            right = torch.zeros_like(x)
            if float(abs(left_denom).item()) > 0.0:
                left = ((x - t[i]) / left_denom) * B[:, i]
            if float(abs(right_denom).item()) > 0.0:
                right = ((t[i + p + 1] - x) / right_denom) * B[:, i + 1]
            cols.append(left + right)
        B = torch.stack(cols, dim=1)
    B = B[:, :num_basis]

    # Open B-splines form a partition of unity including both closed endpoints.
    atol = 1e-12 if x.dtype == torch.float64 else 1e-6
    right_mask = torch.isclose(x, torch.ones_like(x), atol=atol, rtol=0.0)
    left_mask = torch.isclose(x, -torch.ones_like(x), atol=atol, rtol=0.0)
    if torch.any(right_mask):
        B = B.clone()
        B[right_mask] = 0.0
        B[right_mask, -1] = 1.0
    if torch.any(left_mask):
        B = B.clone()
        B[left_mask] = 0.0
        B[left_mask, 0] = 1.0
    return B


class SplineControls(ControlModel):
    """Tensor-product B-spline controls on ``[-1,1]^d``.

    Knots are fixed, open, and uniformly spaced. Only spline coefficients are
    trainable. The tensor product is intentionally explicit: in the low-input-
    dimensional regime of the first LPFN benchmarks it gives the standard
    multivariate spline approximation space and exposes its dimensional cost.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        depth: int,
        num_generators: int,
        num_basis_per_dim: int,
        degree: int = 3,
        init_scale: float = 0.0,
        validate_domain: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if num_basis_per_dim < degree + 1:
            raise ValueError("num_basis_per_dim must be at least degree + 1")
        if not dtype.is_floating_point:
            raise TypeError("Spline coefficients require a floating dtype")
        self.input_dim = int(input_dim)
        self.num_basis_per_dim = int(num_basis_per_dim)
        self.degree = int(degree)
        self.validate_domain = bool(validate_domain)
        self.register_buffer(
            "knots",
            _open_uniform_knots(self.num_basis_per_dim, self.degree, dtype=dtype),
            persistent=True,
        )
        multi = tuple(product(range(self.num_basis_per_dim), repeat=self.input_dim))
        self.register_buffer(
            "multi_indices",
            torch.tensor(multi, dtype=torch.long),
            persistent=True,
        )
        coeff = torch.zeros(
            self.depth,
            self.num_generators,
            self.num_basis_functions,
            dtype=dtype,
        )
        if init_scale > 0:
            coeff.normal_(mean=0.0, std=float(init_scale))
        self.coefficients = nn.Parameter(coeff)

    @property
    def num_basis_functions(self) -> int:
        return self.num_basis_per_dim ** self.input_dim

    def basis_1d(self, x: torch.Tensor) -> torch.Tensor:
        return _bspline_basis_1d(x, self.knots, self.degree, self.num_basis_per_dim)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"x must have input_dim={self.input_dim}; received {int(x.shape[1])}"
            )
        if x.is_complex() or not x.dtype.is_floating_point:
            raise TypeError("x must be a real floating tensor")
        if self.validate_domain:
            tol = 1e-12 if x.dtype == torch.float64 else 1e-6
            if torch.any(x < -1.0 - tol) or torch.any(x > 1.0 + tol):
                raise ValueError("SplineControls expects inputs in [-1, 1]")
        per_dim = [self.basis_1d(x[:, j]) for j in range(self.input_dim)]
        indices = self.multi_indices.to(device=x.device)
        phi = torch.ones(
            (int(x.shape[0]), self.num_basis_functions), dtype=x.dtype, device=x.device
        )
        for j, Bj in enumerate(per_dim):
            phi = phi * Bj[:, indices[:, j]]
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.basis(x)
        coeff = self.coefficients.to(dtype=x.dtype, device=x.device)
        theta = torch.einsum("bn,krn->bkr", phi, coeff)
        return self.validate_output(theta, batch_size=int(x.shape[0]))
