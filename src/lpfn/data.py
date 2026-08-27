from __future__ import annotations

from dataclasses import dataclass

import torch

from lpfn.targets import BoxDomain, TargetFamily


@dataclass(frozen=True)
class DatasetSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def sample_uniform_box(
    domain: BoxDomain,
    n: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    low = torch.tensor(domain.low, dtype=dtype)
    high = torch.tensor(domain.high, dtype=dtype)
    u = torch.rand(n, domain.input_dim, generator=generator, dtype=dtype)
    return low + (high - low) * u


def make_uniform_split(
    target: TargetFamily,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> DatasetSplit:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x_train = sample_uniform_box(target.domain, n_train, generator=generator, dtype=dtype)
    x_val = sample_uniform_box(target.domain, n_val, generator=generator, dtype=dtype)
    x_test = sample_uniform_box(target.domain, n_test, generator=generator, dtype=dtype)
    return DatasetSplit(
        x_train=x_train,
        y_train=target(x_train),
        x_val=x_val,
        y_val=target(x_val),
        x_test=x_test,
        y_test=target(x_test),
    )
