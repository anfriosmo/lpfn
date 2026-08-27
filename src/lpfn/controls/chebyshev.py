from __future__ import annotations

from itertools import product

import torch
from torch import nn

from .base import ControlModel


def _total_degree_multi_indices(input_dim: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        alpha
        for alpha in product(range(degree + 1), repeat=input_dim)
        if sum(alpha) <= degree
    )


class ChebyshevControls(ControlModel):
    """Trainable multivariate Chebyshev controls on ``[-1,1]^d``.

    Each scalar control is expanded in a total-degree tensor-product basis

        theta_{ka}(x) = sum_{|alpha| <= D} c_{ka,alpha}
                         prod_j T_{alpha_j}(x_j).

    The total-degree truncation is used rather than the full box truncation to
    keep the first reference implementation compact in several variables.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        depth: int,
        num_generators: int,
        degree: int,
        init_scale: float = 0.0,
        validate_domain: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if not dtype.is_floating_point:
            raise TypeError("Chebyshev coefficients require a floating dtype")

        self.input_dim = int(input_dim)
        self.degree = int(degree)
        self.validate_domain = bool(validate_domain)
        multi_indices = _total_degree_multi_indices(self.input_dim, self.degree)
        self.register_buffer(
            "multi_indices",
            torch.tensor(multi_indices, dtype=torch.long),
            persistent=True,
        )
        coefficients = torch.zeros(
            self.depth,
            self.num_generators,
            len(multi_indices),
            dtype=dtype,
        )
        if init_scale > 0:
            coefficients.normal_(mean=0.0, std=float(init_scale))
        self.coefficients = nn.Parameter(coefficients)

    @property
    def num_basis_functions(self) -> int:
        return int(self.multi_indices.shape[0])

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
            # Tiny tolerance accommodates data produced by arithmetic near endpoints.
            tol = 1e-12 if x.dtype == torch.float64 else 1e-6
            if torch.any(x < -1.0 - tol) or torch.any(x > 1.0 + tol):
                raise ValueError("ChebyshevControls expects inputs in [-1, 1]")

        batch = int(x.shape[0])
        # values[:, j, n] = T_n(x_j)
        values = torch.empty(
            batch,
            self.input_dim,
            self.degree + 1,
            dtype=x.dtype,
            device=x.device,
        )
        values[:, :, 0] = 1.0
        if self.degree >= 1:
            values[:, :, 1] = x
        for n in range(2, self.degree + 1):
            values[:, :, n] = 2.0 * x * values[:, :, n - 1] - values[:, :, n - 2]

        indices = self.multi_indices.to(device=x.device)
        phi = torch.ones(batch, self.num_basis_functions, dtype=x.dtype, device=x.device)
        for j in range(self.input_dim):
            phi = phi * values[:, j, indices[:, j]]
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.basis(x)
        coeff = self.coefficients.to(dtype=x.dtype, device=x.device)
        theta = torch.einsum("bn,krn->bkr", phi, coeff)
        return self.validate_output(theta, batch_size=int(x.shape[0]))
