"""Merge the certified four-family pilot with the new KAN pilot shards."""
from __future__ import annotations

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
OLD = ROOT / "results" / "fourway_pilot"
NEW_DIRS = (
    ROOT / "results" / "kan_x",
    ROOT / "results" / "kan_xz",
    ROOT / "results" / "kan_h",
)
OUT = ROOT / "results" / "fiveway_pilot"
TARGETS = ("x_rotation", "xz_product", "noncommuting_hamiltonian")
MODELS = ("chebyshev", "fourier", "spline", "kan", "mlp")
DEPTHS = (1, 2, 3)
CAPS = (30, 60, 120)
SEEDS = (11, 23, 37)


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _kan_architecture_summary(selections: list[dict[str, str]]) -> list[dict[str, object]]:
    counts: dict[tuple[object, ...], int] = {}
    totals: dict[tuple[str, int, int], int] = {}
    for row in selections:
        if row["control_model"] != "kan":
            continue
        arch = json.loads(row["control_architecture"])
        hidden = arch["hidden_widths"]
        topology = "direct" if not hidden else f"hidden_{int(hidden[0])}"
        key = (
            row["target"], int(row["depth"]), int(row["parameter_cap"]),
            topology, int(arch["num_basis_per_edge"]), int(arch["degree"]),
            int(row["parameter_count"]),
        )
        counts[key] = counts.get(key, 0) + 1
        base = (row["target"], int(row["depth"]), int(row["parameter_cap"]))
        totals[base] = totals.get(base, 0) + 1
    out = []
    for key, n in sorted(counts.items(), key=lambda item: tuple(map(str, item[0]))):
        target, depth, cap, topology, basis, degree, params = key
        total = totals[(target, depth, cap)]
        out.append({
            "target": target, "depth": depth, "parameter_cap": cap,
            "topology": topology, "num_basis_per_edge": basis,
            "degree": degree, "parameter_count": params,
            "n_selected": n, "n_total": total, "selection_rate": n / total,
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oldm = _manifest(OLD)
    newm = [_manifest(d) for d in NEW_DIRS]
    reference = oldm["config"]
    keys = ["depths", "parameter_caps", "seeds", "n_train", "n_val", "n_test", "epochs", "learning_rates"]
    for m in newm:
        for key in keys:
            if m["config"][key] != reference[key]:
                raise ValueError(f"protocol mismatch in {key}")

    candidates = read_csv_rows(OLD / "candidates.csv")
    selections = read_csv_rows(OLD / "selections.csv")
    old_candidate_count = len(candidates)
    old_selection_count = len(selections)
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
    write_csv_rows(OUT / "kan_architecture_summary.csv", _kan_architecture_summary(selections))

    pair_rows = []
    for left, right in combinations(MODELS, 2):
        pair_rows.extend(paired_selection_differences(long_rows, left=left, right=right))
    write_csv_rows(OUT / "all_pairwise_comparisons.csv", pair_rows)

    provenance = {
        "protocol": "same_optimizer_parameter_cap_fiveway",
        "learning_rate": 0.03,
        "reused": {
            "models": ["chebyshev", "fourier", "spline", "mlp"],
            "source": str(OLD.relative_to(ROOT)),
            "candidate_count": old_candidate_count,
            "selection_count": old_selection_count,
        },
        "new": {
            "models": ["kan"],
            "sources": [str(d.relative_to(ROOT)) for d in NEW_DIRS],
            "candidate_count": len(candidates) - old_candidate_count,
            "selection_count": len(selections) - old_selection_count,
            "search_grid": {
                "hidden_widths": [0, 1, 2], "basis_sizes": [1, 2, 4],
                "degrees": [0, 1, 3],
            },
        },
        "merged": {"candidate_count": len(candidates), "selection_count": len(selections)},
        "pairing": {
            "targets": list(TARGETS), "depths": list(DEPTHS), "parameter_caps": list(CAPS),
            "seeds": list(SEEDS), "models": list(MODELS),
        },
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "protocol": "same_optimizer_parameter_cap_fiveway",
        "derived_from": [str(OLD.relative_to(ROOT)), *[str(d.relative_to(ROOT)) for d in NEW_DIRS]],
        "config": {
            "target_names": list(TARGETS), "control_models": list(MODELS),
            "depths": list(DEPTHS), "parameter_caps": list(CAPS), "seeds": list(SEEDS),
            "n_train": reference["n_train"], "n_val": reference["n_val"], "n_test": reference["n_test"],
            "epochs": reference["epochs"], "learning_rates": reference["learning_rates"],
        },
    }, indent=2), encoding="utf-8")

    generate_pilot_report(OUT, targets=TARGETS, depths=DEPTHS, caps=CAPS, seeds=SEEDS, models=MODELS)
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
