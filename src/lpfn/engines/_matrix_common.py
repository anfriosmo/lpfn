from __future__ import annotations

import torch

from lpfn.generators import GeneratorSet


def validate_theta(theta: torch.Tensor, generators: GeneratorSet) -> None:
    if theta.ndim != 3:
        raise ValueError("theta must have shape [batch, K, r]")
    if theta.is_complex():
        raise TypeError("theta must be real-valued")
    if not theta.dtype.is_floating_point:
        raise TypeError("theta must use a floating dtype")
    if int(theta.shape[-1]) != generators.num_generators:
        raise ValueError(
            "theta's last dimension must equal the number of generators"
        )


def complex_dtype_for(theta: torch.Tensor) -> torch.dtype:
    return torch.complex128 if theta.dtype == torch.float64 else torch.complex64


def prepare_base_point(
    base_point: torch.Tensor | None,
    *,
    batch_size: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    validate: bool = True,
) -> torch.Tensor:
    if base_point is None:
        U0 = torch.eye(n, dtype=dtype, device=device)
        return U0.unsqueeze(0).expand(batch_size, -1, -1).clone()

    if not isinstance(base_point, torch.Tensor):
        raise TypeError("base_point must be a torch.Tensor or None")
    U0 = base_point.to(device=device, dtype=dtype)
    if U0.ndim == 2:
        if tuple(U0.shape) != (n, n):
            raise ValueError(f"base_point must have shape [{n},{n}]")
        U0 = U0.unsqueeze(0).expand(batch_size, -1, -1).clone()
    elif U0.ndim == 3:
        if tuple(U0.shape[1:]) != (n, n):
            raise ValueError(f"batched base_point must end in [{n},{n}]")
        if U0.shape[0] == 1:
            U0 = U0.expand(batch_size, -1, -1).clone()
        elif U0.shape[0] != batch_size:
            raise ValueError("batched base_point must match theta batch size")
    else:
        raise ValueError("base_point must have shape [N,N] or [batch,N,N]")

    if validate:
        identity = torch.eye(n, dtype=dtype, device=device).expand(batch_size, -1, -1)
        defect = U0.mH @ U0 - identity
        atol = 1e-10 if dtype == torch.complex128 else 1e-6
        if not torch.allclose(
            defect, torch.zeros_like(defect), atol=atol, rtol=1e-6
        ):
            raise ValueError("base_point must be unitary in a U(N) matrix engine")
    return U0
