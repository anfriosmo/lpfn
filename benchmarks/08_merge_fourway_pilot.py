"""Merge prior Chebyshev/MLP pilot with new Fourier/Spline shards.

No training is performed here. The script checks protocol-defining fields,
concatenates raw candidate/selection tables, regenerates aggregate reports, and
records provenance for the reused and new computations.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

from lpfn.benchmarking.reporting import (
    audit_selection_output,
    generate_pilot_report,
    read_csv_rows,
    write_csv_rows,
)
from lpfn.benchmarking.selection_runner import (
    _selection_long_rows,
    paired_selection_differences,
    selection_profiles,
    summarize_selections,
)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "selection_pilot"
NEW_DIRS = (
    ROOT / "results" / "fourier_spline_x",
    ROOT / "results" / "fourier_spline_xz",
    ROOT / "results" / "fourier_spline_h",
)
OUT = ROOT / "results" / "fourway_pilot"
TARGETS = ("x_rotation", "xz_product", "noncommuting_hamiltonian")
MODELS = ("chebyshev", "fourier", "spline", "mlp")
DEPTHS = (1, 2, 3)
CAPS = (30, 60, 120)
SEEDS = (11, 23, 37)


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oldm = _manifest(OLD)
    newm = [_manifest(d) for d in NEW_DIRS]
    # Fields that must be identical for pairing to be scientifically valid.
    keys = ["depths", "parameter_caps", "seeds", "n_train", "n_val", "n_test", "epochs", "learning_rates"]
    reference = oldm["config"]
    for m in newm:
        for key in keys:
            if m["config"][key] != reference[key]:
                raise ValueError(f"protocol mismatch in {key}: {m['config'][key]} != {reference[key]}")

    candidates = read_csv_rows(OLD / "candidates.csv")
    selections = read_csv_rows(OLD / "selections.csv")
    for d in NEW_DIRS:
        candidates.extend(read_csv_rows(d / "candidates.csv"))
        selections.extend(read_csv_rows(d / "selections.csv"))

    if len({r["candidate_id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate candidate_id after merge")
    if len({r["selection_id"] for r in selections}) != len(selections):
        raise ValueError("duplicate selection_id after merge")

    write_csv_rows(OUT / "candidates.csv", candidates)
    write_csv_rows(OUT / "selections.csv", selections)
    long_rows = _selection_long_rows(selections)
    write_csv_rows(OUT / "selections_long.csv", long_rows)
    write_csv_rows(OUT / "summary_long.csv", summarize_selections(long_rows))
    write_csv_rows(OUT / "selection_profiles.csv", selection_profiles(selections))

    pair_rows = []
    for left, right in combinations(MODELS, 2):
        pair_rows.extend(paired_selection_differences(long_rows, left=left, right=right))
    write_csv_rows(OUT / "all_pairwise_comparisons.csv", pair_rows)

    provenance = {
        "protocol": "same_optimizer_parameter_cap_fourway",
        "learning_rate": 0.03,
        "reused": {
            "models": ["chebyshev", "mlp"],
            "source": str(OLD.relative_to(ROOT)),
            "candidate_count": len(read_csv_rows(OLD / "candidates.csv")),
            "selection_count": len(read_csv_rows(OLD / "selections.csv")),
        },
        "new": {
            "models": ["fourier", "spline"],
            "sources": [str(d.relative_to(ROOT)) for d in NEW_DIRS],
            "candidate_count": sum(len(read_csv_rows(d / "candidates.csv")) for d in NEW_DIRS),
            "selection_count": sum(len(read_csv_rows(d / "selections.csv")) for d in NEW_DIRS),
        },
        "merged": {"candidate_count": len(candidates), "selection_count": len(selections)},
        "pairing": {
            "targets": list(TARGETS), "depths": list(DEPTHS), "parameter_caps": list(CAPS),
            "seeds": list(SEEDS), "models": list(MODELS),
        },
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Recreate a compact manifest for the merged derived dataset.
    merged_manifest = {
        "protocol": "same_optimizer_parameter_cap_fourway",
        "derived_from": [str(OLD.relative_to(ROOT)), *[str(d.relative_to(ROOT)) for d in NEW_DIRS]],
        "config": {
            "target_names": list(TARGETS), "control_models": list(MODELS),
            "depths": list(DEPTHS), "parameter_caps": list(CAPS), "seeds": list(SEEDS),
            "n_train": reference["n_train"], "n_val": reference["n_val"], "n_test": reference["n_test"],
            "epochs": reference["epochs"], "learning_rates": reference["learning_rates"],
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")

    generate_pilot_report(
        OUT, targets=TARGETS, depths=DEPTHS, caps=CAPS, seeds=SEEDS, models=MODELS
    )
    audit = audit_selection_output(
        OUT, expected_targets=TARGETS, expected_depths=DEPTHS,
        expected_caps=CAPS, expected_seeds=SEEDS, expected_models=MODELS,
    )
    if not audit["passed"]:
        raise RuntimeError(audit)
    print(json.dumps(provenance["merged"], indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
