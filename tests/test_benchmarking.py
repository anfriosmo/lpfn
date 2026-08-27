import csv
import json
from pathlib import Path

import torch

from lpfn.benchmarking import (
    BenchmarkConfig,
    build_control_for_budget,
    chebyshev_parameter_count,
    one_hidden_mlp_parameter_count,
    paired_differences,
    run_benchmark,
    summarize_long,
)


def test_parameter_count_formulas_match_constructed_models():
    cheb, cheb_spec = build_control_for_budget(
        "chebyshev",
        input_dim=2,
        depth=2,
        num_generators=3,
        budget=60,
        seed=4,
    )
    assert cheb.trainable_parameter_count() == cheb_spec.parameter_count
    assert cheb_spec.parameter_count == chebyshev_parameter_count(
        input_dim=2, depth=2, num_generators=3, degree=cheb.degree
    )

    mlp, mlp_spec = build_control_for_budget(
        "mlp",
        input_dim=2,
        depth=2,
        num_generators=3,
        budget=60,
        seed=4,
    )
    width = mlp.hidden_widths[0]
    assert mlp.trainable_parameter_count() == mlp_spec.parameter_count
    assert mlp_spec.parameter_count == one_hidden_mlp_parameter_count(
        input_dim=2, output_dim=6, width=width
    )


def test_budget_matching_is_deterministic():
    a, sa = build_control_for_budget(
        "mlp", input_dim=2, depth=3, num_generators=3, budget=100, seed=19
    )
    b, sb = build_control_for_budget(
        "mlp", input_dim=2, depth=3, num_generators=3, budget=100, seed=19
    )
    assert sa == sb
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_summary_and_pairing_use_seed_matched_observations():
    rows = []
    for seed, c, m in [(1, 0.2, 0.3), (2, 0.4, 0.1)]:
        base = {
            "target": "t",
            "depth": 1,
            "parameter_budget": 30,
            "seed": seed,
            "metric": "frobenius_loss",
        }
        rows.append({**base, "control_model": "chebyshev", "value": c})
        rows.append({**base, "control_model": "mlp", "value": m})

    summary = summarize_long(rows)
    cheb = next(r for r in summary if r["control_model"] == "chebyshev")
    assert cheb["n"] == 2
    assert abs(cheb["mean"] - 0.3) < 1e-15

    paired = paired_differences(rows)
    assert len(paired) == 1
    assert paired[0]["n_pairs"] == 2
    # differences are -0.1 and +0.3 => mean +0.1
    assert abs(paired[0]["mean_difference"] - 0.1) < 1e-15
    assert paired[0]["left_win_rate"] == 0.5


def test_tiny_benchmark_writes_reproducibility_artifacts(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "bench"
    config = BenchmarkConfig(
        target_names=("x_rotation",),
        depths=(1,),
        parameter_budgets=(18,),
        seeds=(3, 5),
        n_train=12,
        n_val=6,
        n_test=10,
        epochs=3,
        learning_rate=0.02,
        save_histories=True,
    )
    paths = run_benchmark(config, root=root, output_dir=out, progress=False)
    for path in paths.values():
        assert path.exists()

    with paths["runs"].open(newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    assert len(runs) == 4  # 2 seeds x 2 control models
    for seed in ("3", "5"):
        same_seed = [r for r in runs if r["seed"] == seed]
        assert {r["control_model"] for r in same_seed} == {"chebyshev", "mlp"}
        assert {r["data_seed"] for r in same_seed} == {seed}

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["config"]["seeds"] == [3, 5]
    assert len(manifest["source"]["tree_sha256"]) == 64
    assert len(list((out / "histories").glob("*.json"))) == 4


def test_benchmark_resume_skips_completed_runs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "resume"
    config = BenchmarkConfig(
        target_names=("x_rotation",),
        control_models=("chebyshev",),
        depths=(1,),
        parameter_budgets=(12,),
        seeds=(7,),
        n_train=8,
        n_val=4,
        n_test=6,
        epochs=2,
        save_histories=False,
    )
    first = run_benchmark(config, root=root, output_dir=out, progress=False)
    before = first["runs"].read_text(encoding="utf-8")
    second = run_benchmark(config, root=root, output_dir=out, progress=False, resume=True)
    after = second["runs"].read_text(encoding="utf-8")
    assert before == after


def test_resume_postprocessing_keeps_numeric_group_keys_unique(tmp_path: Path):
    from lpfn.benchmarking import postprocess_runs

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "typed_resume"
    config = BenchmarkConfig(
        target_names=("x_rotation",),
        depths=(1,),
        parameter_budgets=(30,),
        seeds=(1, 2),
        n_train=8,
        n_val=4,
        n_test=6,
        epochs=2,
        save_histories=False,
    )
    run_benchmark(config, root=root, output_dir=out, progress=False)
    postprocess_runs(out / "runs.csv")
    with (out / "summary_long.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    frob = [r for r in rows if r["metric"] == "frobenius_loss"]
    assert len(frob) == 2  # one row per control model, not duplicated numeric/string groups
