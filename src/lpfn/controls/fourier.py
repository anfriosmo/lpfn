from __future__ import annotations

from itertools import product
import math

import torch
from torch import nn

from .base import ControlModel


def _canonical_frequency_vectors(input_dim: int, max_frequency: int) -> tuple[tuple[int, ...], ...]:
    """Represent nonzero integer frequencies modulo k ~ -k.

    We keep vectors with L1 norm <= max_frequency and choose the representative
    whose first nonzero component is positive. Together with sine and cosine,
    these vectors span the corresponding real multivariate trigonometric
    polynomial space.
    """
    if input_dim < 1:
        raise ValueError("input_dim must be at least 1")
    if max_frequency < 0:
        raise ValueError("max_frequency must be nonnegative")
    out: list[tuple[int, ...]] = []
    rng = range(-max_frequency, max_frequency + 1)
    for k in product(rng, repeat=input_dim):
        if all(v == 0 for v in k) or sum(abs(v) for v in k) > max_frequency:
            continue
        first = next(v for v in k if v != 0)
        if first > 0:
            out.append(tuple(int(v) for v in k))
    out.sort(key=lambda k: (sum(abs(v) for v in k), k))
    return tuple(out)


class FourierControls(ControlModel):
    """Real multivariate Fourier controls on ``[-1,1]^d``.

    The period is 2 in each coordinate. With canonical frequency vectors k,

        theta(x) = c0 + sum_k [a_k cos(pi k.x) + b_k sin(pi k.x)].

    ``max_frequency`` uses an L1 truncation, so mixed modes are included and
    the model is not restricted to an additive Fourier ansatz.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        depth: int,
        num_generators: int,
        max_frequency: int,
        init_scale: float = 0.0,
        validate_domain: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if max_frequency < 0:
            raise ValueError("max_frequency must be nonnegative")
        if not dtype.is_floating_point:
            raise TypeError("Fourier coefficients require a floating dtype")
        self.input_dim = int(input_dim)
        self.max_frequency = int(max_frequency)
        self.validate_domain = bool(validate_domain)
        frequencies = _canonical_frequency_vectors(self.input_dim, self.max_frequency)
        if frequencies:
            freq_tensor = torch.tensor(frequencies, dtype=torch.long)
        else:
            freq_tensor = torch.empty((0, self.input_dim), dtype=torch.long)
        self.register_buffer("frequencies", freq_tensor, persistent=True)
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
    def num_frequency_vectors(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def num_basis_functions(self) -> int:
        return 1 + 2 * self.num_frequency_vectors

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
                raise ValueError("FourierControls expects inputs in [-1, 1]")

        batch = int(x.shape[0])
        ones = torch.ones((batch, 1), dtype=x.dtype, device=x.device)
        if self.num_frequency_vectors == 0:
            return ones
        freq = self.frequencies.to(device=x.device, dtype=x.dtype)
        phase = math.pi * (x @ freq.T)
        return torch.cat((ones, torch.cos(phase), torch.sin(phase)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.basis(x)
        coeff = self.coefficients.to(dtype=x.dtype, device=x.device)
        theta = torch.einsum("bn,krn->bkr", phi, coeff)
        return self.validate_output(theta, batch_size=int(x.shape[0]))
