from __future__ import annotations

import csv
from pathlib import Path

from lpfn.benchmarking.reporting import (
    aggregate_selection_metric,
    audit_selection_output,
    best_mean_cells,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_audit_rejects_neither_valid_caps_nor_candidate_table(tmp_path: Path) -> None:
    _write(tmp_path / 'candidates.csv', [{
        'candidate_id': 'c1', 'parameter_count': 3, 'parameter_cap': 5,
        'target': 't', 'control_model': 'chebyshev',
    }])
    _write(tmp_path / 'selections.csv', [{
        'selection_id': 's1', 'max_unitarity_defect': 1e-15,
        'target': 't', 'control_model': 'chebyshev', 'depth': 1,
        'parameter_cap': 5, 'seed': 1,
    }])
    audit = audit_selection_output(
        tmp_path,
        expected_targets=('t',), expected_depths=(1,), expected_caps=(5,),
        expected_seeds=(1,), expected_models=('chebyshev',),
    )
    assert audit['passed'] is True
    assert audit['cap_violation_count'] == 0
    assert audit['candidate_test_columns'] == []


def test_best_mean_cells_uses_groupwise_minimum() -> None:
    rows = [
        {'target': 'a', 'control_model': 'm', 'mean': 2.0, 'depth': 1},
        {'target': 'a', 'control_model': 'm', 'mean': 1.0, 'depth': 2},
        {'target': 'a', 'control_model': 'n', 'mean': 3.0, 'depth': 1},
    ]
    best = best_mean_cells(rows)
    assert len(best) == 2
    assert next(r for r in best if r['control_model'] == 'm')['depth'] == 2


def test_aggregate_selection_metric_statistics() -> None:
    rows = [
        {'target':'a','depth':'1','parameter_cap':'5','control_model':'m','frobenius_loss':'1.0'},
        {'target':'a','depth':'1','parameter_cap':'5','control_model':'m','frobenius_loss':'3.0'},
    ]
    out = aggregate_selection_metric(rows, metric='frobenius_loss')
    assert out[0]['mean'] == 2.0
    assert out[0]['n'] == 2

from lpfn.benchmarking.reporting import (
    aggregate_protocol_comparison,
    compare_selection_protocols,
    learning_rate_selection_summary,
)


def test_compare_selection_protocols_pairs_exact_cell_and_delta() -> None:
    fixed = [{
        'selection_id':'s','target':'t','control_model':'m','depth':'1','parameter_cap':'5','seed':'7',
        'learning_rate':'0.03','parameter_count':'3','frobenius_loss':'2.0','selected_candidate_id':'a',
    }]
    tuned = [{
        'selection_id':'s','target':'t','control_model':'m','depth':'1','parameter_cap':'5','seed':'7',
        'learning_rate':'0.01','parameter_count':'4','frobenius_loss':'1.5','selected_candidate_id':'b',
    }]
    rows = compare_selection_protocols(fixed, tuned)
    assert rows[0]['tuned_minus_fixed'] == -0.5
    assert rows[0]['tuned_improved'] is True
    assert rows[0]['selected_candidate_changed'] is True
    agg = aggregate_protocol_comparison(rows)
    assert agg[0]['tuned_test_improvement_rate'] == 1.0
    assert agg[0]['nondefault_learning_rate_rate'] == 1.0


def test_learning_rate_selection_summary_rates_sum_to_one() -> None:
    rows = [
        {'target':'t','control_model':'m','learning_rate':'0.03'},
        {'target':'t','control_model':'m','learning_rate':'0.01'},
        {'target':'t','control_model':'m','learning_rate':'0.03'},
    ]
    out = learning_rate_selection_summary(rows)
    assert sum(float(r['selection_rate']) for r in out) == 1.0
