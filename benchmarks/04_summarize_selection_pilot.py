"""Generate paper-facing summaries from a completed nested-selection pilot."""
from __future__ import annotations

from pathlib import Path

from lpfn.benchmarking.reporting import generate_pilot_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "selection_pilot"


def main() -> None:
    paths = generate_pilot_report(
        OUTPUT,
        targets=("x_rotation", "xz_product", "noncommuting_hamiltonian"),
        depths=(1, 2, 3),
        caps=(30, 60, 120),
        seeds=(11, 23, 37),
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
