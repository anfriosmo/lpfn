"""Export an auditable edge-function decomposition for the best linear KAN cell.

The xz_product K=2, cap=30 cell is especially useful because validation selects
one direct degree-1 KAN layer in all three paired seeds.  A two-basis linear
B-spline edge is exactly affine, so its slope/intercept can be recovered from
its two coefficients without curve fitting.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from lpfn import LieProductNetwork, PauliGeneratorSet, PauliMatrixEngine
from lpfn.benchmarking.config import ControlSpec
from lpfn.benchmarking.factory import build_control_from_spec
from lpfn.benchmarking.reporting import read_csv_rows, write_csv_rows

ROOT = Path(__file__).resolve().parents[1]
SHARD = ROOT / "results" / "kan_xz"
OUT = ROOT / "results" / "fiveway_pilot"
INPUT_LABELS = ("x", "y")
GENERATOR_LABELS = ("X", "Y", "Z")


def main() -> None:
    selections = read_csv_rows(SHARD / "selections.csv")
    candidates = {r["candidate_id"]: r for r in read_csv_rows(SHARD / "candidates.csv")}
    selected = [
        r for r in selections
        if int(r["depth"]) == 2 and int(r["parameter_cap"]) == 30
    ]
    rows: list[dict[str, object]] = []
    for selection in selected:
        cand = candidates[selection["selected_candidate_id"]]
        arch = json.loads(cand["control_architecture"])
        if arch["hidden_widths"] or int(arch["degree"]) != 1 or int(arch["num_basis_per_edge"]) != 2:
            raise RuntimeError("expected the selected K=2/cap=30 KAN to be direct linear B2")
        spec = ControlSpec(
            kind="kan", parameter_budget=30,
            parameter_count=int(cand["parameter_count"]), architecture=arch,
        )
        controls = build_control_from_spec(
            spec, input_dim=2, depth=2, num_generators=3,
            seed=int(selection["seed"]),
        )
        model = LieProductNetwork(
            generators=PauliGeneratorSet(1, include_identity=False),
            controls=controls, engine=PauliMatrixEngine(),
        )
        state = torch.load(SHARD / cand["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        coeff = controls.effective_coefficients(0).detach()
        # For the open degree-1 basis with two functions:
        # B0=(1-t)/2, B1=(1+t)/2.
        slope = (coeff[:, :, 1] - coeff[:, :, 0]) / 2.0
        intercept = (coeff[:, :, 0] + coeff[:, :, 1]) / 2.0
        for input_index, input_label in enumerate(INPUT_LABELS):
            for flat_output in range(6):
                block = flat_output // 3
                generator = GENERATOR_LABELS[flat_output % 3]
                rows.append({
                    "seed": int(selection["seed"]),
                    "test_frobenius_loss": float(selection["frobenius_loss"]),
                    "input": input_label,
                    "block": block + 1,
                    "generator": generator,
                    "slope": float(slope[input_index, flat_output]),
                    "intercept": float(intercept[input_index, flat_output]),
                    "absolute_slope": abs(float(slope[input_index, flat_output])),
                })
    write_csv_rows(OUT / "kan_xz_linear_edges.csv", rows)

    # Aggregate the affine decomposition across paired seeds.
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["input"]), int(row["block"]), str(row["generator"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (input_label, block, generator), values in sorted(groups.items()):
        slopes = torch.tensor([float(v["slope"]) for v in values], dtype=torch.float64)
        ints = torch.tensor([float(v["intercept"]) for v in values], dtype=torch.float64)
        summary.append({
            "input": input_label, "block": block, "generator": generator,
            "mean_slope": float(slopes.mean()),
            "std_slope": float(slopes.std(unbiased=True)),
            "mean_intercept": float(ints.mean()),
            "mean_absolute_slope": float(slopes.abs().mean()),
        })
    write_csv_rows(OUT / "kan_xz_linear_edges_summary.csv", summary)
    print(OUT / "kan_xz_linear_edges.csv")
    print(OUT / "kan_xz_linear_edges_summary.csv")


if __name__ == "__main__":
    main()
