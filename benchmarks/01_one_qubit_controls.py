"""Controlled first benchmark: ChebyshevControls vs MLPControls.

This is a reproducibility smoke benchmark, not a tuned leaderboard.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from lpfn import (
    ChebyshevControls,
    LieProductNetwork,
    MLPControls,
    PauliGeneratorSet,
    PauliMatrixEngine,
    Trainer,
    XRotationTarget,
    XZProductTarget,
    NoncommutingHamiltonianTarget,
    evaluate_unitary_model,
    make_uniform_split,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "one_qubit_smoke.csv"


def build_controls(kind: str, target_name: str, input_dim: int, depth: int, r: int):
    if kind == "chebyshev":
        degree = 5 if input_dim == 1 else 3
        return ChebyshevControls(
            input_dim=input_dim,
            depth=depth,
            num_generators=r,
            degree=degree,
            init_scale=0.02,
        )
    if kind == "mlp":
        if target_name == "x_rotation":
            widths = (4,)
        elif target_name == "xz_product":
            widths = (6,)
        else:
            widths = (7,)
        return MLPControls(
            input_dim=input_dim,
            depth=depth,
            num_generators=r,
            hidden_widths=widths,
            activation="tanh",
        )
    raise ValueError(kind)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    targets = [XRotationTarget(), XZProductTarget(), NoncommutingHamiltonianTarget()]
    depths = {
        "x_rotation": 1,
        "xz_product": 2,
        "noncommuting_hamiltonian": 3,
    }
    generators = PauliGeneratorSet(1, include_identity=False)
    rows: list[dict[str, object]] = []

    for target in targets:
        split = make_uniform_split(
            target,
            n_train=128,
            n_val=64,
            n_test=256,
            seed=20260823,
        )
        for kind in ("chebyshev", "mlp"):
            torch.manual_seed(20260823)
            depth = depths[target.name]
            controls = build_controls(
                kind, target.name, target.input_dim, depth, generators.num_generators
            )
            model = LieProductNetwork(
                generators=generators,
                controls=controls,
                engine=PauliMatrixEngine(),
            )
            trainer = Trainer(
                epochs=800,
                learning_rate=0.03,
                seed=20260823,
            )
            result = trainer.fit(
                model,
                x_train=split.x_train,
                y_train=split.y_train,
                x_val=split.x_val,
                y_val=split.y_val,
            )
            metrics = evaluate_unitary_model(model, split.x_test, split.y_test)
            row = {
                "target": target.name,
                "control_model": kind,
                "depth": depth,
                "parameters": model.trainable_parameter_count(),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "train_seconds": result.wall_time_seconds,
                **metrics,
            }
            rows.append(row)
            print(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
