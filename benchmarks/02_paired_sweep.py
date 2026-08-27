"""Paired, budget-controlled LPFN benchmark.

Examples
--------
Quick infrastructure validation:
    PYTHONPATH=src python benchmarks/02_paired_sweep.py --preset validation

Substantive pilot (multiple seeds, budgets and residual depths):
    PYTHONPATH=src python benchmarks/02_paired_sweep.py --preset pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lpfn.benchmarking import BenchmarkConfig, run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def config_for_preset(name: str) -> BenchmarkConfig:
    targets = ("x_rotation", "xz_product", "noncommuting_hamiltonian")
    if name == "validation":
        return BenchmarkConfig(
            target_names=targets,
            depths=(1, 2),
            parameter_budgets=(30, 60),
            seeds=(11, 23),
            n_train=64,
            n_val=32,
            n_test=128,
            epochs=80,
            learning_rate=0.03,
            save_histories=True,
        )
    if name == "pilot":
        return BenchmarkConfig(
            target_names=targets,
            depths=(1, 2, 3),
            parameter_budgets=(30, 60, 120),
            seeds=(11, 23, 37),
            n_train=128,
            n_val=64,
            n_test=256,
            epochs=250,
            learning_rate=0.03,
            save_histories=True,
        )
    if name == "paper":
        return BenchmarkConfig(
            target_names=targets,
            depths=(1, 2, 3, 4),
            parameter_budgets=(30, 60, 120, 240),
            seeds=(11, 23, 37, 51, 73, 89, 101, 131, 167, 197),
            n_train=256,
            n_val=128,
            n_test=1024,
            epochs=1200,
            learning_rate=0.03,
            save_histories=True,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("validation", "pilot", "paper"), default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = config_for_preset(args.preset)
    output = args.output or (ROOT / "results" / f"paired_{args.preset}")
    paths = run_benchmark(
        config,
        root=ROOT,
        output_dir=output,
        command=[sys.executable, *sys.argv],
        resume=args.resume,
    )
    print("\nOutputs")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
