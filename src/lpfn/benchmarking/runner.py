from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from lpfn import (
    LieProductNetwork,
    NoncommutingHamiltonianTarget,
    PauliGeneratorSet,
    PauliMatrixEngine,
    Trainer,
    XRotationTarget,
    XZProductTarget,
    evaluate_unitary_model,
    make_uniform_split,
)
from lpfn.targets import TargetFamily

from .config import BenchmarkConfig
from .factory import build_control_for_budget
from .manifest import build_manifest, write_manifest


TARGETS: dict[str, type[TargetFamily]] = {
    "x_rotation": XRotationTarget,
    "xz_product": XZProductTarget,
    "noncommuting_hamiltonian": NoncommutingHamiltonianTarget,
}

PRIMARY_METRICS = (
    "frobenius_loss",
    "phase_loss",
    "mean_operator_error",
    "max_operator_error",
    "mean_phase_fidelity",
    "max_unitarity_defect",
    "train_seconds",
    "inference_seconds",
    "best_val_loss",
)

# +1 means higher is better; -1 means lower is better.

RUN_INT_FIELDS = {
    "depth", "parameter_budget", "parameter_count", "budget_absolute_error",
    "seed", "data_seed", "init_seed", "n_train", "n_val", "n_test",
    "epochs", "best_epoch",
}
RUN_FLOAT_FIELDS = {
    "budget_relative_error", "learning_rate", "best_val_loss", "train_seconds",
    "frobenius_loss", "phase_loss", "mean_operator_error", "max_operator_error",
    "mean_phase_fidelity", "max_unitarity_defect", "inference_seconds",
}


def _coerce_run_row(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    for key in RUN_INT_FIELDS:
        if key in out and out[key] != "":
            out[key] = int(out[key])
    for key in RUN_FLOAT_FIELDS:
        if key in out and out[key] != "":
            out[key] = float(out[key])
    return out

METRIC_DIRECTION = {
    "mean_phase_fidelity": +1,
    "frobenius_loss": -1,
    "phase_loss": -1,
    "mean_operator_error": -1,
    "max_operator_error": -1,
    "max_unitarity_defect": -1,
    "train_seconds": -1,
    "inference_seconds": -1,
    "best_val_loss": -1,
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _long_rows(wide_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    id_keys = [
        "run_id",
        "target",
        "control_model",
        "depth",
        "parameter_budget",
        "parameter_count",
        "budget_relative_error",
        "seed",
        "data_seed",
        "init_seed",
    ]
    for row in wide_rows:
        base = {key: row[key] for key in id_keys}
        for metric in PRIMARY_METRICS:
            if metric in row:
                out.append({**base, "metric": metric, "value": row[metric]})
    return out


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize_long(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in long_rows:
        key = (
            row["target"],
            row["control_model"],
            row["depth"],
            row["parameter_budget"],
            row["metric"],
        )
        groups[key].append(float(row["value"]))

    summary: list[dict[str, object]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        target, control, depth, budget, metric = key
        n = len(values)
        std = _sample_std(values)
        summary.append(
            {
                "target": target,
                "control_model": control,
                "depth": depth,
                "parameter_budget": budget,
                "metric": metric,
                "n": n,
                "mean": statistics.fmean(values),
                "std": std,
                "sem": std / math.sqrt(n) if n else float("nan"),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return summary


def paired_differences(long_rows: list[dict[str, object]], *, left: str = "chebyshev", right: str = "mlp") -> list[dict[str, object]]:
    indexed: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in long_rows:
        key = (
            row["target"],
            row["depth"],
            row["parameter_budget"],
            row["seed"],
            row["metric"],
        )
        indexed[key][str(row["control_model"])] = float(row["value"])

    grouped: dict[tuple[object, ...], list[tuple[float, float, float]]] = defaultdict(list)
    for key, values in indexed.items():
        if left not in values or right not in values:
            continue
        target, depth, budget, seed, metric = key
        lval = values[left]
        rval = values[right]
        grouped[(target, depth, budget, metric)].append((lval - rval, lval, rval))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        target, depth, budget, metric = key
        diffs = [v[0] for v in values]
        n = len(diffs)
        std = _sample_std(diffs)
        direction = METRIC_DIRECTION.get(str(metric), -1)
        left_wins = 0
        ties = 0
        for _, lval, rval in values:
            if math.isclose(lval, rval, rel_tol=1e-12, abs_tol=1e-15):
                ties += 1
            elif (lval > rval and direction > 0) or (lval < rval and direction < 0):
                left_wins += 1
        rows.append(
            {
                "target": target,
                "depth": depth,
                "parameter_budget": budget,
                "metric": metric,
                "left_model": left,
                "right_model": right,
                "difference_definition": f"{left}-{right}",
                "n_pairs": n,
                "mean_difference": statistics.fmean(diffs),
                "std_difference": std,
                "sem_difference": std / math.sqrt(n) if n else float("nan"),
                "median_difference": statistics.median(diffs),
                "left_win_rate": left_wins / n if n else float("nan"),
                "tie_rate": ties / n if n else float("nan"),
            }
        )
    return rows


def postprocess_runs(runs_path: Path, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Regenerate long-form, summary, and paired tables from an existing runs.csv."""
    output_dir = output_dir or runs_path.parent
    with runs_path.open(newline="", encoding="utf-8") as f:
        wide_rows = [_coerce_run_row(dict(row)) for row in csv.DictReader(f)]
    long_rows = _long_rows(wide_rows)
    summary = summarize_long(long_rows)
    paired = paired_differences(long_rows)
    _write_csv(output_dir / "runs_long.csv", long_rows)
    _write_csv(output_dir / "summary_long.csv", summary)
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    return {
        "runs_long": output_dir / "runs_long.csv",
        "summary": output_dir / "summary_long.csv",
        "paired": output_dir / "paired_comparisons.csv",
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    root: Path,
    output_dir: Path,
    command: list[str] | None = None,
    progress: bool = True,
    resume: bool = False,
) -> dict[str, Path]:
    torch.set_default_dtype(torch.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "histories"
    if config.save_histories:
        history_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    runs_path = output_dir / "runs.csv"
    wide_rows: list[dict[str, object]] = []
    if resume and runs_path.exists():
        with runs_path.open(newline="", encoding="utf-8") as f:
            wide_rows = [_coerce_run_row(dict(row)) for row in csv.DictReader(f)]
    elif runs_path.exists() and not resume:
        raise FileExistsError(
            f"{runs_path} already exists; use resume=True or choose a new output directory"
        )

    if resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("config") != config.to_dict():
            raise ValueError("cannot resume: benchmark configuration differs from existing manifest")
    else:
        manifest = build_manifest(root=root, config=config.to_dict(), command=command)
        write_manifest(manifest_path, manifest)

    completed_run_ids = {str(row["run_id"]) for row in wide_rows}
    initial_run_count = len(wide_rows)
    generators = PauliGeneratorSet(1, include_identity=False)
    total = (
        len(config.target_names)
        * len(config.depths)
        * len(config.parameter_budgets)
        * len(config.seeds)
        * len(config.control_models)
    )
    completed = len(wide_rows)
    benchmark_start = perf_counter()

    for target_name in config.target_names:
        if target_name not in TARGETS:
            raise ValueError(f"unknown target {target_name!r}; choose from {sorted(TARGETS)}")
        target = TARGETS[target_name]()
        for depth in config.depths:
            for budget in config.parameter_budgets:
                for seed in config.seeds:
                    # Paired design: same split for all control models in the cell.
                    data_seed = int(seed)
                    split = make_uniform_split(
                        target,
                        n_train=config.n_train,
                        n_val=config.n_val,
                        n_test=config.n_test,
                        seed=data_seed,
                    )
                    for kind in config.control_models:
                        run_id = f"{target_name}__K{depth}__B{budget}__S{seed}__{kind}"
                        if run_id in completed_run_ids:
                            if progress:
                                print(f"[{completed:>3}/{total}] skip {run_id}")
                            continue
                        init_seed = int(seed)
                        controls, spec = build_control_for_budget(
                            kind,
                            input_dim=target.input_dim,
                            depth=depth,
                            num_generators=generators.num_generators,
                            budget=budget,
                            seed=init_seed,
                            chebyshev_init_scale=config.chebyshev_init_scale,
                            mlp_activation=config.mlp_activation,
                        )
                        model = LieProductNetwork(
                            generators=generators,
                            controls=controls,
                            engine=PauliMatrixEngine(),
                        )
                        trainer = Trainer(
                            epochs=config.epochs,
                            learning_rate=config.learning_rate,
                            seed=init_seed,
                        )
                        result = trainer.fit(
                            model,
                            x_train=split.x_train,
                            y_train=split.y_train,
                            x_val=split.x_val,
                            y_val=split.y_val,
                        )
                        metrics = evaluate_unitary_model(model, split.x_test, split.y_test)
                        row: dict[str, object] = {
                            "run_id": run_id,
                            "target": target_name,
                            "control_model": kind,
                            "depth": depth,
                            "parameter_budget": budget,
                            "parameter_count": spec.parameter_count,
                            "budget_absolute_error": spec.absolute_budget_error,
                            "budget_relative_error": spec.relative_budget_error,
                            "control_architecture": json.dumps(spec.architecture, sort_keys=True),
                            "seed": seed,
                            "data_seed": data_seed,
                            "init_seed": init_seed,
                            "n_train": config.n_train,
                            "n_val": config.n_val,
                            "n_test": config.n_test,
                            "epochs": config.epochs,
                            "learning_rate": config.learning_rate,
                            "best_epoch": result.best_epoch,
                            "best_val_loss": result.best_val_loss,
                            "train_seconds": result.wall_time_seconds,
                            **metrics,
                        }
                        wide_rows.append(row)
                        completed_run_ids.add(run_id)
                        # Crash-safe enough for research runs: persist the completed table
                        # after every run, so long sweeps can be resumed without recomputation.
                        _write_csv(runs_path, wide_rows)
                        if config.save_histories:
                            (history_dir / f"{run_id}.json").write_text(
                                json.dumps(
                                    {
                                        "run_id": run_id,
                                        "train_loss": result.train_loss,
                                        "val_loss": result.val_loss,
                                        "best_epoch": result.best_epoch,
                                        "best_val_loss": result.best_val_loss,
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        completed += 1
                        if progress:
                            print(
                                f"[{completed:>3}/{total}] {run_id} "
                                f"params={spec.parameter_count} test={metrics['frobenius_loss']:.6g}"
                            )

    long_rows = _long_rows(wide_rows)
    summary = summarize_long(long_rows)
    paired = paired_differences(long_rows)

    _write_csv(runs_path, wide_rows)
    _write_csv(output_dir / "runs_long.csv", long_rows)
    _write_csv(output_dir / "summary_long.csv", summary)
    _write_csv(output_dir / "paired_comparisons.csv", paired)

    elapsed = perf_counter() - benchmark_start
    total_recorded_train_seconds = sum(float(row["train_seconds"]) for row in wide_rows)
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "runs": len(wide_rows),
                "new_runs_this_invocation": len(wide_rows) - initial_run_count,
                "last_invocation_wall_time_seconds": elapsed,
                "sum_recorded_train_seconds_all_runs": total_recorded_train_seconds,
                "note": "Wall time is per invocation; use per-run train_seconds for cross-run timing analysis.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "runs": runs_path,
        "runs_long": output_dir / "runs_long.csv",
        "summary": output_dir / "summary_long.csv",
        "paired": output_dir / "paired_comparisons.csv",
        "manifest": manifest_path,
        "completed": output_dir / "COMPLETED.json",
    }
