from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_selection_output(
    output_dir: Path,
    *,
    expected_targets: Iterable[str] | None = None,
    expected_depths: Iterable[int] | None = None,
    expected_caps: Iterable[int] | None = None,
    expected_seeds: Iterable[int] | None = None,
    expected_models: Iterable[str] | None = None,
) -> dict[str, object]:
    """Audit a completed nested-selection benchmark without retraining anything."""
    candidates = read_csv_rows(output_dir / "candidates.csv")
    selections = read_csv_rows(output_dir / "selections.csv")

    candidate_ids = [r["candidate_id"] for r in candidates]
    selection_ids = [r["selection_id"] for r in selections]
    cap_violations = [
        r["candidate_id"]
        for r in candidates
        if int(r["parameter_count"]) > int(r["parameter_cap"])
    ]
    candidate_test_columns = sorted(
        key for key in (candidates[0].keys() if candidates else []) if "test" in key.lower()
    )

    expected_selection_count = None
    if all(x is not None for x in (
        expected_targets, expected_depths, expected_caps, expected_seeds, expected_models
    )):
        expected_selection_count = (
            len(tuple(expected_targets or ()))
            * len(tuple(expected_depths or ()))
            * len(tuple(expected_caps or ()))
            * len(tuple(expected_seeds or ()))
            * len(tuple(expected_models or ()))
        )

    max_unitarity = max(
        (float(r["max_unitarity_defect"]) for r in selections),
        default=float("nan"),
    )
    audit = {
        "candidate_count": len(candidates),
        "unique_candidate_count": len(set(candidate_ids)),
        "selection_count": len(selections),
        "unique_selection_count": len(set(selection_ids)),
        "expected_selection_count": expected_selection_count,
        "cap_violation_count": len(cap_violations),
        "cap_violations": cap_violations,
        "candidate_test_columns": candidate_test_columns,
        "max_unitarity_defect": max_unitarity,
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "selection_ids_unique": len(selection_ids) == len(set(selection_ids)),
        "selection_count_matches_expected": (
            expected_selection_count is None or len(selections) == expected_selection_count
        ),
        "no_cap_violations": not cap_violations,
        "candidate_table_has_no_test_metrics": not candidate_test_columns,
    }
    audit["passed"] = all([
        audit["candidate_ids_unique"],
        audit["selection_ids_unique"],
        audit["selection_count_matches_expected"],
        audit["no_cap_violations"],
        audit["candidate_table_has_no_test_metrics"],
    ])
    return audit


def aggregate_selection_metric(
    selections: list[dict[str, str]],
    *,
    metric: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in selections:
        key = (
            row["target"], int(row["depth"]), int(row["parameter_cap"]), row["control_model"]
        )
        groups[key].append(float(row[metric]))

    out: list[dict[str, object]] = []
    for (target, depth, cap, model), values in sorted(groups.items()):
        n = len(values)
        std = statistics.stdev(values) if n >= 2 else 0.0
        out.append({
            "target": target,
            "depth": depth,
            "parameter_cap": cap,
            "control_model": model,
            "metric": metric,
            "n": n,
            "mean": statistics.fmean(values),
            "std": std,
            "sem": std / math.sqrt(n) if n else float("nan"),
            "min": min(values),
            "max": max(values),
        })
    return out


def aggregate_selected_complexity(
    selections: list[dict[str, str]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, str], list[tuple[float, float]]] = defaultdict(list)
    for row in selections:
        key = (
            row["target"], int(row["depth"]), int(row["parameter_cap"]), row["control_model"]
        )
        groups[key].append((float(row["parameter_count"]), float(row["selected_parameter_fraction"])))

    out: list[dict[str, object]] = []
    for (target, depth, cap, model), values in sorted(groups.items()):
        counts = [v[0] for v in values]
        fracs = [v[1] for v in values]
        out.append({
            "target": target,
            "depth": depth,
            "parameter_cap": cap,
            "control_model": model,
            "n": len(values),
            "mean_parameter_count": statistics.fmean(counts),
            "min_parameter_count": min(counts),
            "max_parameter_count": max(counts),
            "mean_cap_fraction": statistics.fmean(fracs),
            "min_cap_fraction": min(fracs),
            "max_cap_fraction": max(fracs),
        })
    return out


def best_mean_cells(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[str, str], dict[str, object]] = {}
    for row in metric_rows:
        key = (str(row["target"]), str(row["control_model"]))
        if key not in best or float(row["mean"]) < float(best[key]["mean"]):
            best[key] = row
    return [best[k] for k in sorted(best)]


def generate_pilot_report(
    output_dir: Path,
    *,
    targets: tuple[str, ...],
    depths: tuple[int, ...],
    caps: tuple[int, ...],
    seeds: tuple[int, ...],
    models: tuple[str, ...] = ("chebyshev", "mlp"),
) -> dict[str, Path]:
    selections = read_csv_rows(output_dir / "selections.csv")
    audit = audit_selection_output(
        output_dir,
        expected_targets=targets,
        expected_depths=depths,
        expected_caps=caps,
        expected_seeds=seeds,
        expected_models=models,
    )
    frob = aggregate_selection_metric(selections, metric="frobenius_loss")
    complexity = aggregate_selected_complexity(selections)
    best = best_mean_cells(frob)

    audit_path = output_dir / "AUDIT.json"
    frob_path = output_dir / "paper_frobenius_summary.csv"
    complexity_path = output_dir / "paper_selected_complexity.csv"
    best_path = output_dir / "paper_best_cells.csv"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_csv_rows(frob_path, frob)
    write_csv_rows(complexity_path, complexity)
    write_csv_rows(best_path, best)
    return {
        "audit": audit_path,
        "frobenius_summary": frob_path,
        "selected_complexity": complexity_path,
        "best_cells": best_path,
    }


def learning_rate_selection_summary(
    selections: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Summarize which validation-selected learning rates survive to test."""
    groups: dict[tuple[str, str, float], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in selections:
        target = row["target"]
        model = row["control_model"]
        lr = float(row["learning_rate"])
        groups[(target, model, lr)] += 1
        totals[(target, model)] += 1
    out: list[dict[str, object]] = []
    for (target, model, lr), count in sorted(groups.items()):
        total = totals[(target, model)]
        out.append({
            "target": target,
            "control_model": model,
            "learning_rate": lr,
            "n_selected": count,
            "n_total": total,
            "selection_rate": count / total,
        })
    return out


def compare_selection_protocols(
    fixed: list[dict[str, str]],
    tuned: list[dict[str, str]],
    *,
    metric: str = "frobenius_loss",
) -> list[dict[str, object]]:
    """Pair fixed-LR and tuned-LR selections by the exact benchmark cell."""
    fixed_by = {row["selection_id"]: row for row in fixed}
    tuned_by = {row["selection_id"]: row for row in tuned}
    if set(fixed_by) != set(tuned_by):
        missing_fixed = sorted(set(tuned_by) - set(fixed_by))
        missing_tuned = sorted(set(fixed_by) - set(tuned_by))
        raise ValueError(
            f"protocol selection ids differ; missing_fixed={missing_fixed[:3]}, "
            f"missing_tuned={missing_tuned[:3]}"
        )
    out: list[dict[str, object]] = []
    for sid in sorted(fixed_by):
        a, b = fixed_by[sid], tuned_by[sid]
        av, bv = float(a[metric]), float(b[metric])
        out.append({
            "selection_id": sid,
            "target": b["target"],
            "control_model": b["control_model"],
            "depth": int(b["depth"]),
            "parameter_cap": int(b["parameter_cap"]),
            "seed": int(b["seed"]),
            "metric": metric,
            "fixed_learning_rate": float(a["learning_rate"]),
            "tuned_learning_rate": float(b["learning_rate"]),
            "fixed_parameter_count": int(a["parameter_count"]),
            "tuned_parameter_count": int(b["parameter_count"]),
            "fixed_value": av,
            "tuned_value": bv,
            "tuned_minus_fixed": bv - av,
            "tuned_improved": bv < av,
            "selected_candidate_changed": a["selected_candidate_id"] != b["selected_candidate_id"],
        })
    return out


def aggregate_protocol_comparison(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate tuned-vs-fixed deltas by target and control model."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["target"]), str(row["control_model"]))].append(row)
    out: list[dict[str, object]] = []
    for (target, model), vals in sorted(groups.items()):
        deltas = [float(r["tuned_minus_fixed"]) for r in vals]
        n = len(vals)
        changed = sum(bool(r["selected_candidate_changed"]) for r in vals)
        improved = sum(bool(r["tuned_improved"]) for r in vals)
        nondefault_lr = sum(float(r["tuned_learning_rate"]) != float(r["fixed_learning_rate"]) for r in vals)
        std = statistics.stdev(deltas) if n >= 2 else 0.0
        out.append({
            "target": target,
            "control_model": model,
            "n_cells": n,
            "mean_tuned_minus_fixed": statistics.fmean(deltas),
            "std_tuned_minus_fixed": std,
            "sem_tuned_minus_fixed": std / math.sqrt(n) if n else float("nan"),
            "median_tuned_minus_fixed": statistics.median(deltas),
            "tuned_test_improvement_rate": improved / n if n else float("nan"),
            "selected_candidate_change_rate": changed / n if n else float("nan"),
            "nondefault_learning_rate_rate": nondefault_lr / n if n else float("nan"),
        })
    return out


def generate_optimizer_protocol_comparison(
    fixed_dir: Path,
    tuned_dir: Path,
    *,
    output_dir: Path | None = None,
    metric: str = "frobenius_loss",
) -> dict[str, Path]:
    """Generate auditable fixed-vs-tuned optimizer comparison tables."""
    output_dir = output_dir or tuned_dir
    fixed = read_csv_rows(fixed_dir / "selections.csv")
    tuned = read_csv_rows(tuned_dir / "selections.csv")
    pair_rows = compare_selection_protocols(fixed, tuned, metric=metric)
    agg = aggregate_protocol_comparison(pair_rows)
    lr = learning_rate_selection_summary(tuned)
    pair_path = output_dir / "tuned_vs_fixed_by_selection.csv"
    agg_path = output_dir / "tuned_vs_fixed_summary.csv"
    lr_path = output_dir / "learning_rate_selection_summary.csv"
    write_csv_rows(pair_path, pair_rows)
    write_csv_rows(agg_path, agg)
    write_csv_rows(lr_path, lr)
    return {"by_selection": pair_path, "summary": agg_path, "learning_rates": lr_path}
