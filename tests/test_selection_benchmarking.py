import csv
import json
from pathlib import Path

import pytest

from lpfn.benchmarking import (
    SelectionBenchmarkConfig,
    admissible_control_specs,
    run_selection_benchmark,
)


def test_admissible_specs_are_hard_capped_and_include_simpler_models():
    cheb = admissible_control_specs(
        "chebyshev",
        input_dim=1,
        depth=1,
        num_generators=3,
        parameter_cap=30,
        chebyshev_degrees=(0, 1, 2, 3, 4, 5, 6),
        mlp_widths=(1, 2, 4),
    )
    assert all(s.parameter_count <= 30 for s in cheb)
    degrees = [int(s.architecture["degree"]) for s in cheb]
    assert 0 in degrees and 1 in degrees
    assert len(degrees) > 1

    mlp = admissible_control_specs(
        "mlp",
        input_dim=1,
        depth=1,
        num_generators=3,
        parameter_cap=30,
        chebyshev_degrees=(0, 1),
        mlp_widths=(1, 2, 4, 8, 16),
    )
    assert all(s.parameter_count <= 30 for s in mlp)
    assert len(mlp) >= 2


def _tiny_config() -> SelectionBenchmarkConfig:
    return SelectionBenchmarkConfig(
        target_names=("x_rotation",),
        depths=(1,),
        parameter_caps=(30,),
        seeds=(3,),
        n_train=12,
        n_val=6,
        n_test=10,
        epochs=3,
        learning_rates=(0.01, 0.02),
        chebyshev_degrees=(0, 1, 2),
        mlp_widths=(1, 2, 4),
        save_histories=True,
        save_checkpoints=True,
    )


def test_nested_selection_uses_validation_and_candidates_have_no_test_metrics(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    paths = run_selection_benchmark(
        _tiny_config(), root=root, output_dir=tmp_path / "nested", progress=False
    )
    with paths["candidates"].open(newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    with paths["selections"].open(newline="", encoding="utf-8") as f:
        selections = list(csv.DictReader(f))

    assert len(selections) == 2  # one final Chebyshev and one final MLP
    assert candidates
    for row in candidates:
        assert int(row["parameter_count"]) <= int(row["parameter_cap"])
        for test_metric in (
            "frobenius_loss",
            "phase_loss",
            "mean_operator_error",
            "max_operator_error",
            "mean_phase_fidelity",
            "max_unitarity_defect",
            "inference_seconds",
        ):
            assert test_metric not in row

    for selected in selections:
        same_cell = [
            r for r in candidates
            if r["target"] == selected["target"]
            and r["control_model"] == selected["control_model"]
            and r["depth"] == selected["depth"]
            and r["parameter_cap"] == selected["parameter_cap"]
            and r["seed"] == selected["seed"]
        ]
        expected = min(
            same_cell,
            key=lambda r: (
                float(r["best_val_loss"]),
                int(r["parameter_count"]),
                r["candidate_id"],
            ),
        )
        assert selected["selected_candidate_id"] == expected["candidate_id"]
        assert int(selected["parameter_count"]) <= int(selected["parameter_cap"])


def test_test_evaluation_is_called_once_per_final_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import lpfn.benchmarking.selection_runner as sr

    calls = 0
    original = sr.evaluate_unitary_model

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sr, "evaluate_unitary_model", counted)
    root = Path(__file__).resolve().parents[1]
    run_selection_benchmark(
        _tiny_config(), root=root, output_dir=tmp_path / "counted", progress=False
    )
    assert calls == 2  # exactly one test evaluation per control model final cell


def test_nested_resume_skips_candidates_and_test_re_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import lpfn.benchmarking.selection_runner as sr

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "resume"
    config = _tiny_config()
    first = run_selection_benchmark(config, root=root, output_dir=out, progress=False)
    before_candidates = first["candidates"].read_text(encoding="utf-8")
    before_selections = first["selections"].read_text(encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("test evaluation should not repeat for completed selections")

    monkeypatch.setattr(sr, "evaluate_unitary_model", fail_if_called)
    run_selection_benchmark(config, root=root, output_dir=out, progress=False, resume=True)
    assert before_candidates == first["candidates"].read_text(encoding="utf-8")
    assert before_selections == first["selections"].read_text(encoding="utf-8")


def test_manifest_describes_no_test_selection_protocol(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    paths = run_selection_benchmark(
        _tiny_config(), root=root, output_dir=tmp_path / "manifest", progress=False
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["protocol"]["name"] == "nested_validation_parameter_cap"
    assert "test" in manifest["protocol"]["test_policy"].lower()


def test_selection_profiles_report_cap_usage(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    paths = run_selection_benchmark(
        _tiny_config(), root=root, output_dir=tmp_path / "profiles", progress=False
    )
    with paths["selection_profiles"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        count = int(row["parameter_count"])
        cap = int(row["parameter_cap"])
        assert count <= cap
        assert abs(float(row["selected_parameter_fraction"]) - count / cap) < 1e-12
        assert 0.0 < float(row["selection_rate"]) <= 1.0


def test_fourier_and_spline_specs_are_hard_capped_and_constructible():
    from lpfn.benchmarking import build_control_from_spec

    for kind in ("fourier", "spline"):
        specs = admissible_control_specs(
            kind,
            input_dim=2,
            depth=1,
            num_generators=3,
            parameter_cap=120,
            fourier_max_frequencies=(0, 1, 2, 3, 4),
            spline_basis_sizes=(1, 2, 3, 4, 5, 6, 8),
            spline_degrees=(0, 1, 2, 3),
        )
        assert specs
        assert all(s.parameter_count <= 120 for s in specs)
        for spec in specs:
            model = build_control_from_spec(
                spec, input_dim=2, depth=1, num_generators=3, seed=1
            )
            assert model.trainable_parameter_count() == spec.parameter_count


def test_fourway_nested_selection_runs_without_test_leakage(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = SelectionBenchmarkConfig(
        target_names=("x_rotation",),
        control_models=("chebyshev", "fourier", "spline", "mlp"),
        depths=(1,), parameter_caps=(30,), seeds=(5,),
        n_train=12, n_val=6, n_test=10, epochs=2,
        learning_rates=(0.02,),
        chebyshev_degrees=(0, 1),
        fourier_max_frequencies=(0, 1),
        spline_basis_sizes=(1, 2, 3, 4, 5), spline_degrees=(0, 1, 2, 3),
        mlp_widths=(1, 2),
    )
    paths = run_selection_benchmark(
        config, root=root, output_dir=tmp_path / "fourway", progress=False
    )
    with paths["candidates"].open(newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    with paths["selections"].open(newline="", encoding="utf-8") as f:
        selections = list(csv.DictReader(f))
    assert {r["control_model"] for r in selections} == {"chebyshev", "fourier", "spline", "mlp"}
    assert len(selections) == 4
    assert all(int(r["parameter_count"]) <= int(r["parameter_cap"]) for r in candidates)
    assert all("frobenius_loss" not in r for r in candidates)
