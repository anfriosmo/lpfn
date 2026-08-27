"""Nested validation benchmark under hard parameter caps.

Examples
--------
Infrastructure validation:
    PYTHONPATH=src python benchmarks/03_nested_parameter_cap.py --preset validation

Small scientific pilot:
    PYTHONPATH=src python benchmarks/03_nested_parameter_cap.py --preset pilot

The candidate table never contains test metrics. Architecture and (optionally)
learning rate are selected by validation loss; test is evaluated once per final
selected model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lpfn.benchmarking import SelectionBenchmarkConfig, run_selection_benchmark

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("x_rotation", "xz_product", "noncommuting_hamiltonian")


def config_for_preset(name: str) -> SelectionBenchmarkConfig:
    if name == "validation":
        return SelectionBenchmarkConfig(
            target_names=("x_rotation",),
            depths=(1,),
            parameter_caps=(30,),
            seeds=(11, 23),
            n_train=32,
            n_val=16,
            n_test=32,
            epochs=25,
            learning_rates=(0.03,),
            chebyshev_degrees=(0, 1, 2, 3),
            mlp_widths=(1, 2, 4, 8),
        )
    if name == "pilot":
        return SelectionBenchmarkConfig(
            target_names=TARGETS,
            depths=(1, 2, 3),
            parameter_caps=(30, 60, 120),
            seeds=(11, 23, 37),
            n_train=128,
            n_val=64,
            n_test=256,
            epochs=250,
            learning_rates=(0.03,),  # same-optimizer benchmark
            chebyshev_degrees=tuple(range(0, 9)),
            mlp_widths=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
        )
    if name == "tuned-pilot":
        return SelectionBenchmarkConfig(
            target_names=TARGETS,
            depths=(1, 2, 3),
            parameter_caps=(30, 60, 120),
            seeds=(11, 23, 37),
            n_train=128,
            n_val=64,
            n_test=256,
            epochs=250,
            learning_rates=(0.003, 0.01, 0.03),
            chebyshev_degrees=tuple(range(0, 9)),
            mlp_widths=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
        )
    if name == "paper":
        return SelectionBenchmarkConfig(
            target_names=TARGETS,
            depths=(1, 2, 3, 4),
            parameter_caps=(30, 60, 120, 240),
            seeds=(11, 23, 37, 51, 73, 89, 101, 131, 167, 197),
            n_train=256,
            n_val=128,
            n_test=1024,
            epochs=1200,
            learning_rates=(0.003, 0.01, 0.03),
            chebyshev_degrees=tuple(range(0, 13)),
            mlp_widths=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128),
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("validation", "pilot", "tuned-pilot", "paper"),
        default="validation",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = config_for_preset(args.preset)
    output = args.output or (ROOT / "results" / f"selection_{args.preset}")
    paths = run_selection_benchmark(
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
