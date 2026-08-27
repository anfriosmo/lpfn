from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
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

from .config import ControlSpec
from .factory import admissible_control_specs, build_control_from_spec
from .manifest import build_manifest, write_manifest
from .selection_config import SelectionBenchmarkConfig


TARGETS: dict[str, type[TargetFamily]] = {
    "x_rotation": XRotationTarget,
    "xz_product": XZProductTarget,
    "noncommuting_hamiltonian": NoncommutingHamiltonianTarget,
}

TEST_METRICS = (
    "frobenius_loss",
    "phase_loss",
    "mean_operator_error",
    "max_operator_error",
    "mean_phase_fidelity",
    "max_unitarity_defect",
    "inference_seconds",
)
SELECTION_METRICS = (*TEST_METRICS, "best_val_loss", "train_seconds", "selected_parameter_fraction")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _arch_token(spec: ControlSpec) -> str:
    if spec.kind == "chebyshev":
        return f"D{int(spec.architecture['degree'])}"
    if spec.kind == "fourier":
        return f"F{int(spec.architecture['max_frequency'])}"
    if spec.kind == "spline":
        return f"S{int(spec.architecture['num_basis_per_dim'])}P{int(spec.architecture['degree'])}"
    if spec.kind == "kan":
        hidden = spec.architecture["hidden_widths"]
        htoken = "D" if not hidden else f"H{int(hidden[0])}"
        return (f"{htoken}B{int(spec.architecture['num_basis_per_edge'])}"
                f"P{int(spec.architecture['degree'])}")
    if spec.kind == "mlp":
        return f"W{int(spec.architecture['hidden_widths'][0])}"
    raise ValueError(spec.kind)


def _lr_token(lr: float) -> str:
    return f"{float(lr):.8g}".replace("-", "m").replace(".", "p")


def _candidate_id(
    *, target: str, depth: int, cap: int, seed: int, kind: str, spec: ControlSpec, lr: float
) -> str:
    return (
        f"{target}__K{depth}__C{cap}__S{seed}__{kind}__"
        f"{_arch_token(spec)}__LR{_lr_token(lr)}"
    )


def _selection_id(*, target: str, depth: int, cap: int, seed: int, kind: str) -> str:
    return f"{target}__K{depth}__C{cap}__S{seed}__{kind}"


def _train_candidate(
    model: LieProductNetwork,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    """Train/rank a candidate using *only* train and validation tensors.

    Deliberately no test tensors are accepted by this function. This is a
    structural guard against test-set leakage during architecture selection.
    """
    trainer = Trainer(epochs=epochs, learning_rate=learning_rate, seed=seed)
    return trainer.fit(
        model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )


def _candidate_specs_for_kind(
    config: SelectionBenchmarkConfig,
    *,
    kind: str,
    target: TargetFamily,
    depth: int,
    num_generators: int,
    cap: int,
) -> list[ControlSpec]:
    return admissible_control_specs(
        kind,
        input_dim=target.input_dim,
        depth=depth,
        num_generators=num_generators,
        parameter_cap=cap,
        chebyshev_degrees=config.chebyshev_degrees,
        fourier_max_frequencies=config.fourier_max_frequencies,
        spline_basis_sizes=config.spline_basis_sizes,
        spline_degrees=config.spline_degrees,
        kan_hidden_widths=config.kan_hidden_widths,
        kan_basis_sizes=config.kan_basis_sizes,
        kan_degrees=config.kan_degrees,
        mlp_widths=config.mlp_widths,
        mlp_activation=config.mlp_activation,
    )


def _selection_long_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    id_keys = (
        "selection_id",
        "selected_candidate_id",
        "target",
        "control_model",
        "depth",
        "parameter_cap",
        "parameter_count",
        "seed",
        "data_seed",
    )
    for row in rows:
        base = {k: row[k] for k in id_keys}
        for metric in SELECTION_METRICS:
            if metric in row:
                out.append({**base, "metric": metric, "value": float(row[metric])})
    return out


def summarize_selections(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in long_rows:
        key = (
            row["target"], row["control_model"], int(row["depth"]),
            int(row["parameter_cap"]), row["metric"],
        )
        groups[key].append(float(row["value"]))
    out: list[dict[str, object]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        target, control, depth, cap, metric = key
        n = len(values)
        std = statistics.stdev(values) if n >= 2 else 0.0
        out.append({
            "target": target,
            "control_model": control,
            "depth": depth,
            "parameter_cap": cap,
            "metric": metric,
            "n": n,
            "mean": statistics.fmean(values),
            "std": std,
            "sem": std / math.sqrt(n) if n else float("nan"),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        })
    return out


_METRIC_DIRECTION = {"mean_phase_fidelity": +1}


def paired_selection_differences(
    long_rows: list[dict[str, object]], *, left: str = "chebyshev", right: str = "mlp"
) -> list[dict[str, object]]:
    indexed: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in long_rows:
        key = (
            row["target"], int(row["depth"]), int(row["parameter_cap"]),
            int(row["seed"]), row["metric"],
        )
        indexed[key][str(row["control_model"])] = float(row["value"])

    groups: dict[tuple[object, ...], list[tuple[float, float, float]]] = defaultdict(list)
    for key, vals in indexed.items():
        if left in vals and right in vals:
            target, depth, cap, _seed, metric = key
            groups[(target, depth, cap, metric)].append(
                (vals[left] - vals[right], vals[left], vals[right])
            )

    out: list[dict[str, object]] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        target, depth, cap, metric = key
        diffs = [v[0] for v in vals]
        n = len(diffs)
        std = statistics.stdev(diffs) if n >= 2 else 0.0
        direction = _METRIC_DIRECTION.get(str(metric), -1)
        left_wins = 0
        ties = 0
        for _, lval, rval in vals:
            if math.isclose(lval, rval, rel_tol=1e-12, abs_tol=1e-15):
                ties += 1
            elif (direction > 0 and lval > rval) or (direction < 0 and lval < rval):
                left_wins += 1
        out.append({
            "target": target,
            "depth": depth,
            "parameter_cap": cap,
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
        })
    return out


def selection_profiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Count how often validation selects each architecture under each cap."""
    groups: dict[tuple[object, ...], int] = defaultdict(int)
    totals: dict[tuple[object, ...], int] = defaultdict(int)
    for row in rows:
        base = (
            row["target"], row["control_model"], int(row["depth"]), int(row["parameter_cap"]),
        )
        architecture = str(row["control_architecture"])
        lr = float(row["learning_rate"])
        count = int(row["parameter_count"])
        groups[(*base, architecture, lr, count)] += 1
        totals[base] += 1

    out: list[dict[str, object]] = []
    for key, n_selected in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        target, control, depth, cap, architecture, lr, count = key
        total = totals[(target, control, depth, cap)]
        out.append({
            "target": target,
            "control_model": control,
            "depth": depth,
            "parameter_cap": cap,
            "parameter_count": count,
            "selected_parameter_fraction": count / cap,
            "control_architecture": architecture,
            "learning_rate": lr,
            "n_selected": n_selected,
            "n_total": total,
            "selection_rate": n_selected / total,
        })
    return out


def run_selection_benchmark(
    config: SelectionBenchmarkConfig,
    *,
    root: Path,
    output_dir: Path,
    command: list[str] | None = None,
    progress: bool = True,
    resume: bool = False,
) -> dict[str, Path]:
    """Run nested train/validation selection under a hard parameter cap.

    Candidate models are never evaluated on test. Test evaluation occurs once,
    after validation has selected a candidate for a final benchmark cell.
    """
    torch.set_default_dtype(torch.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = output_dir / "candidate_histories"
    checkpoints_dir = output_dir / "candidate_checkpoints"
    if config.save_histories:
        histories_dir.mkdir(parents=True, exist_ok=True)
    if config.save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    candidates_path = output_dir / "candidates.csv"
    selections_path = output_dir / "selections.csv"

    candidate_rows = _read_csv(candidates_path) if resume else []
    selection_rows = _read_csv(selections_path) if resume else []
    if not resume and (candidates_path.exists() or selections_path.exists()):
        raise FileExistsError(
            "selection benchmark output already exists; use resume=True or a new directory"
        )

    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config") != config.to_dict():
            raise ValueError("cannot resume: selection benchmark config differs from manifest")
    else:
        manifest = build_manifest(root=root, config=config.to_dict(), command=command)
        manifest["protocol"] = {
            "name": "nested_validation_parameter_cap",
            "selection_rule": "minimum restored best validation Frobenius loss",
            "test_policy": "evaluate test exactly once for one validation-selected candidate per final cell",
        }
        write_manifest(manifest_path, manifest)

    candidate_by_id = {str(r["candidate_id"]): r for r in candidate_rows}
    selection_ids = {str(r["selection_id"]) for r in selection_rows}
    generators = PauliGeneratorSet(1, include_identity=False)
    start = perf_counter()
    new_candidates = 0
    new_selections = 0

    for target_name in config.target_names:
        if target_name not in TARGETS:
            raise ValueError(f"unknown target {target_name!r}")
        target = TARGETS[target_name]()
        for depth in config.depths:
            for cap in config.parameter_caps:
                for seed in config.seeds:
                    data_seed = int(seed)
                    split = make_uniform_split(
                        target,
                        n_train=config.n_train,
                        n_val=config.n_val,
                        n_test=config.n_test,
                        seed=data_seed,
                    )
                    for kind in config.control_models:
                        specs = _candidate_specs_for_kind(
                            config,
                            kind=kind,
                            target=target,
                            depth=depth,
                            num_generators=generators.num_generators,
                            cap=cap,
                        )
                        cell_candidate_ids: list[str] = []
                        spec_lookup: dict[str, ControlSpec] = {}
                        lr_lookup: dict[str, float] = {}

                        for spec in specs:
                            assert spec.parameter_count <= cap
                            for lr in config.learning_rates:
                                cid = _candidate_id(
                                    target=target_name,
                                    depth=depth,
                                    cap=cap,
                                    seed=seed,
                                    kind=kind,
                                    spec=spec,
                                    lr=lr,
                                )
                                cell_candidate_ids.append(cid)
                                spec_lookup[cid] = spec
                                lr_lookup[cid] = float(lr)
                                if cid in candidate_by_id:
                                    continue

                                model = LieProductNetwork(
                                    generators=generators,
                                    controls=build_control_from_spec(
                                        spec,
                                        input_dim=target.input_dim,
                                        depth=depth,
                                        num_generators=generators.num_generators,
                                        seed=int(seed),
                                        chebyshev_init_scale=config.chebyshev_init_scale,
                                        fourier_init_scale=config.fourier_init_scale,
                                        spline_init_scale=config.spline_init_scale,
                                        kan_init_scale=config.kan_init_scale,
                                        mlp_activation=config.mlp_activation,
                                    ),
                                    engine=PauliMatrixEngine(),
                                )
                                result = _train_candidate(
                                    model,
                                    x_train=split.x_train,
                                    y_train=split.y_train,
                                    x_val=split.x_val,
                                    y_val=split.y_val,
                                    epochs=config.epochs,
                                    learning_rate=float(lr),
                                    seed=int(seed),
                                )

                                # Persist the restored-best state before marking the candidate complete.
                                checkpoint_path = checkpoints_dir / f"{cid}.pt"
                                torch.save(model.state_dict(), checkpoint_path)
                                if config.save_histories:
                                    (histories_dir / f"{cid}.json").write_text(
                                        json.dumps({
                                            "candidate_id": cid,
                                            "train_loss": result.train_loss,
                                            "val_loss": result.val_loss,
                                            "best_epoch": result.best_epoch,
                                            "best_val_loss": result.best_val_loss,
                                        }, indent=2),
                                        encoding="utf-8",
                                    )

                                row: dict[str, object] = {
                                    "candidate_id": cid,
                                    "target": target_name,
                                    "control_model": kind,
                                    "depth": depth,
                                    "parameter_cap": cap,
                                    "parameter_count": spec.parameter_count,
                                    "cap_slack": cap - spec.parameter_count,
                                    "control_architecture": json.dumps(spec.architecture, sort_keys=True),
                                    "learning_rate": float(lr),
                                    "seed": seed,
                                    "data_seed": data_seed,
                                    "init_seed": int(seed),
                                    "n_train": config.n_train,
                                    "n_val": config.n_val,
                                    "epochs": config.epochs,
                                    "best_epoch": result.best_epoch,
                                    "best_val_loss": result.best_val_loss,
                                    "train_seconds": result.wall_time_seconds,
                                    "checkpoint": str(checkpoint_path.relative_to(output_dir)),
                                }
                                candidate_rows.append(row)
                                candidate_by_id[cid] = row
                                _write_csv(candidates_path, candidate_rows)
                                new_candidates += 1
                                if progress:
                                    print(
                                        f"candidate {cid}: params={spec.parameter_count}/{cap} "
                                        f"val={result.best_val_loss:.6g}"
                                    )

                        sid = _selection_id(
                            target=target_name, depth=depth, cap=cap, seed=seed, kind=kind
                        )
                        if sid in selection_ids:
                            continue

                        eligible = [candidate_by_id[cid] for cid in cell_candidate_ids]
                        if len(eligible) != len(cell_candidate_ids):
                            raise RuntimeError("not all candidates are complete; cannot select")
                        # Validation only. Simpler model breaks numerically indistinguishable ties.
                        selected = min(
                            eligible,
                            key=lambda r: (
                                float(r["best_val_loss"]),
                                int(r["parameter_count"]),
                                str(r["candidate_id"]),
                            ),
                        )
                        selected_id = str(selected["candidate_id"])
                        selected_spec = spec_lookup[selected_id]
                        selected_lr = lr_lookup[selected_id]
                        model = LieProductNetwork(
                            generators=generators,
                            controls=build_control_from_spec(
                                selected_spec,
                                input_dim=target.input_dim,
                                depth=depth,
                                num_generators=generators.num_generators,
                                seed=int(seed),
                                chebyshev_init_scale=config.chebyshev_init_scale,
                                fourier_init_scale=config.fourier_init_scale,
                                spline_init_scale=config.spline_init_scale,
                                kan_init_scale=config.kan_init_scale,
                                mlp_activation=config.mlp_activation,
                            ),
                            engine=PauliMatrixEngine(),
                        )
                        checkpoint_path = output_dir / str(selected["checkpoint"])
                        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                        model.load_state_dict(state)

                        # The only test-set access in the entire candidate-selection path.
                        test_metrics = evaluate_unitary_model(model, split.x_test, split.y_test)
                        selection_row: dict[str, object] = {
                            "selection_id": sid,
                            "selected_candidate_id": selected_id,
                            "target": target_name,
                            "control_model": kind,
                            "depth": depth,
                            "parameter_cap": cap,
                            "parameter_count": int(selected["parameter_count"]),
                            "cap_slack": cap - int(selected["parameter_count"]),
                            "selected_parameter_fraction": int(selected["parameter_count"]) / cap,
                            "control_architecture": selected["control_architecture"],
                            "learning_rate": selected_lr,
                            "seed": seed,
                            "data_seed": data_seed,
                            "n_train": config.n_train,
                            "n_val": config.n_val,
                            "n_test": config.n_test,
                            "epochs": config.epochs,
                            "candidate_count": len(eligible),
                            "best_epoch": int(selected["best_epoch"]),
                            "best_val_loss": float(selected["best_val_loss"]),
                            "train_seconds": float(selected["train_seconds"]),
                            **test_metrics,
                        }
                        selection_rows.append(selection_row)
                        selection_ids.add(sid)
                        _write_csv(selections_path, selection_rows)
                        new_selections += 1
                        if progress:
                            print(
                                f"SELECT {sid}: {selected_id} -> "
                                f"test={test_metrics['frobenius_loss']:.6g}"
                            )

    long_rows = _selection_long_rows(selection_rows)
    summary = summarize_selections(long_rows)
    paired = paired_selection_differences(long_rows)
    long_path = output_dir / "selections_long.csv"
    summary_path = output_dir / "summary_long.csv"
    paired_path = output_dir / "paired_comparisons.csv"
    profiles_path = output_dir / "selection_profiles.csv"
    profiles = selection_profiles(selection_rows)
    _write_csv(candidates_path, candidate_rows)
    _write_csv(selections_path, selection_rows)
    _write_csv(long_path, long_rows)
    _write_csv(summary_path, summary)
    _write_csv(paired_path, paired)
    _write_csv(profiles_path, profiles)

    completed_path = output_dir / "COMPLETED.json"
    completed_path.write_text(
        json.dumps({
            "candidate_runs": len(candidate_rows),
            "final_selections": len(selection_rows),
            "new_candidates_this_invocation": new_candidates,
            "new_selections_this_invocation": new_selections,
            "wall_time_seconds": perf_counter() - start,
            "test_evaluations_expected": len(selection_rows),
            "protocol": "nested_validation_parameter_cap",
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "candidates": candidates_path,
        "selections": selections_path,
        "selections_long": long_path,
        "summary": summary_path,
        "paired": paired_path,
        "selection_profiles": profiles_path,
        "manifest": manifest_path,
        "completed": completed_path,
    }
