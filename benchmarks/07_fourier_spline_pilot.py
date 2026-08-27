"""Same-optimizer Fourier/Spline extension of the LPFN parameter-cap pilot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lpfn.benchmarking import SelectionBenchmarkConfig, run_selection_benchmark

ROOT = Path(__file__).resolve().parents[1]
ALL_TARGETS = ("x_rotation", "xz_product", "noncommuting_hamiltonian")


def make_config(targets: tuple[str, ...]) -> SelectionBenchmarkConfig:
    return SelectionBenchmarkConfig(
        target_names=targets,
        control_models=("fourier", "spline"),
        depths=(1, 2, 3),
        parameter_caps=(30, 60, 120),
        seeds=(11, 23, 37),
        n_train=128,
        n_val=64,
        n_test=256,
        epochs=250,
        learning_rates=(0.03,),
        fourier_max_frequencies=tuple(range(0, 7)),
        # Compact but heterogeneous grid: validation chooses both resolution
        # and spline degree under the same hard parameter cap.
        spline_basis_sizes=(1, 2, 4, 8),
        spline_degrees=(0, 1, 3),
        save_histories=True,
        save_checkpoints=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=ALL_TARGETS + ("all",), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = ALL_TARGETS if args.target == "all" else (args.target,)
    paths = run_selection_benchmark(
        make_config(targets), root=ROOT, output_dir=args.output,
        command=[sys.executable, *sys.argv], resume=args.resume,
    )
    for k, v in paths.items():
        print(k, v)


if __name__ == "__main__":
    main()
