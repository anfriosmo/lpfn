from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SelectionBenchmarkConfig:
    """Nested validation benchmark with parameter caps rather than quotas."""

    target_names: tuple[str, ...]
    control_models: tuple[str, ...] = ("chebyshev", "mlp")
    depths: tuple[int, ...] = (1, 2, 3)
    parameter_caps: tuple[int, ...] = (30, 60, 120)
    seeds: tuple[int, ...] = (11, 23, 37)
    n_train: int = 128
    n_val: int = 64
    n_test: int = 256
    epochs: int = 500
    learning_rates: tuple[float, ...] = (0.03,)
    chebyshev_degrees: tuple[int, ...] = tuple(range(0, 13))
    fourier_max_frequencies: tuple[int, ...] = tuple(range(0, 9))
    spline_basis_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32)
    spline_degrees: tuple[int, ...] = (0, 1, 2, 3)
    kan_hidden_widths: tuple[int, ...] = (0, 1, 2, 4, 8)
    kan_basis_sizes: tuple[int, ...] = (1, 2, 4, 8)
    kan_degrees: tuple[int, ...] = (0, 1, 3)
    mlp_widths: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    chebyshev_init_scale: float = 0.02
    fourier_init_scale: float = 0.02
    spline_init_scale: float = 0.02
    kan_init_scale: float = 0.02
    mlp_activation: str = "tanh"
    save_histories: bool = True
    save_checkpoints: bool = True

    def __post_init__(self) -> None:
        if not self.target_names:
            raise ValueError("target_names must be non-empty")
        if not self.control_models:
            raise ValueError("control_models must be non-empty")
        allowed = {"chebyshev", "fourier", "spline", "kan", "mlp"}
        unknown = set(self.control_models) - allowed
        if unknown:
            raise ValueError(f"unknown control_models: {sorted(unknown)}")
        if any(k < 1 for k in self.depths):
            raise ValueError("depths must be positive")
        if any(b < 1 for b in self.parameter_caps):
            raise ValueError("parameter_caps must be positive")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if min(self.n_train, self.n_val, self.n_test, self.epochs) < 1:
            raise ValueError("dataset sizes and epochs must be positive")
        if not self.learning_rates or any(lr <= 0 for lr in self.learning_rates):
            raise ValueError("learning_rates must be non-empty and positive")
        if not self.chebyshev_degrees or any(d < 0 for d in self.chebyshev_degrees):
            raise ValueError("chebyshev_degrees must be non-empty and nonnegative")
        if not self.fourier_max_frequencies or any(m < 0 for m in self.fourier_max_frequencies):
            raise ValueError("fourier_max_frequencies must be non-empty and nonnegative")
        if not self.spline_degrees or any(p < 0 for p in self.spline_degrees):
            raise ValueError("spline_degrees must be non-empty and nonnegative")
        if not self.spline_basis_sizes or any(n < 1 for n in self.spline_basis_sizes):
            raise ValueError("spline_basis_sizes must be non-empty and positive")
        if not self.kan_hidden_widths or any(w < 0 for w in self.kan_hidden_widths):
            raise ValueError("kan_hidden_widths must be non-empty and nonnegative")
        if not self.kan_basis_sizes or any(n < 1 for n in self.kan_basis_sizes):
            raise ValueError("kan_basis_sizes must be non-empty and positive")
        if not self.kan_degrees or any(p < 0 for p in self.kan_degrees):
            raise ValueError("kan_degrees must be non-empty and nonnegative")
        if not self.mlp_widths or any(w < 1 for w in self.mlp_widths):
            raise ValueError("mlp_widths must be non-empty and positive")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in (
            "target_names", "control_models", "depths", "parameter_caps", "seeds",
            "learning_rates", "chebyshev_degrees", "fourier_max_frequencies",
            "spline_basis_sizes", "spline_degrees", "kan_hidden_widths",
            "kan_basis_sizes", "kan_degrees", "mlp_widths",
        ):
            data[key] = list(data[key])
        return data
