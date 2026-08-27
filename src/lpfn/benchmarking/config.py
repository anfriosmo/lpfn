from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for paired LPFN control-model benchmarks.

    Seeds are paired across control models: for a fixed target/depth/budget/seed,
    every control model sees exactly the same train/validation/test split.
    """

    target_names: tuple[str, ...]
    control_models: tuple[str, ...] = ("chebyshev", "mlp")
    depths: tuple[int, ...] = (1, 2, 3)
    parameter_budgets: tuple[int, ...] = (30, 60, 120)
    seeds: tuple[int, ...] = (11, 23, 37, 51, 73)
    n_train: int = 128
    n_val: int = 64
    n_test: int = 256
    epochs: int = 500
    learning_rate: float = 0.03
    chebyshev_init_scale: float = 0.02
    mlp_activation: str = "tanh"
    save_histories: bool = True

    def __post_init__(self) -> None:
        if not self.target_names:
            raise ValueError("target_names must be non-empty")
        if not self.control_models:
            raise ValueError("control_models must be non-empty")
        if any(k < 1 for k in self.depths):
            raise ValueError("depths must be positive")
        if any(b < 1 for b in self.parameter_budgets):
            raise ValueError("parameter_budgets must be positive")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if min(self.n_train, self.n_val, self.n_test, self.epochs) < 1:
            raise ValueError("dataset sizes and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        # Keep JSON output stable and explicit.
        for key in (
            "target_names",
            "control_models",
            "depths",
            "parameter_budgets",
            "seeds",
        ):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class ControlSpec:
    kind: str
    parameter_budget: int
    parameter_count: int
    architecture: dict[str, object]

    @property
    def absolute_budget_error(self) -> int:
        return abs(self.parameter_count - self.parameter_budget)

    @property
    def relative_budget_error(self) -> float:
        return self.absolute_budget_error / self.parameter_budget
