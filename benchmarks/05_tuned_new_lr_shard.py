"""Run a target shard for the new tuned-pilot learning rates.

This helper trains only lr in {0.003, 0.01}. The lr=0.03 candidates are
reused from the same-optimizer pilot and all three rates are merged before
final validation selection/test evaluation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from lpfn.benchmarking import SelectionBenchmarkConfig, run_selection_benchmark
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target', required=True, choices=('x_rotation','xz_product','noncommuting_hamiltonian'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a=p.parse_args()
    cfg=SelectionBenchmarkConfig(
        target_names=(a.target,), depths=(1,2,3), parameter_caps=(30,60,120),
        seeds=(11,23,37), n_train=128,n_val=64,n_test=256,epochs=250,
        learning_rates=(0.003,0.01), chebyshev_degrees=tuple(range(0,9)),
        mlp_widths=(1,2,4,8,12,16,24,32,48,64),
    )
    run_selection_benchmark(cfg, root=ROOT, output_dir=a.output,
        command=[sys.executable,*sys.argv], resume=a.resume, progress=False)
if __name__=='__main__': main()
