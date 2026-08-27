"""Generate fixed-learning-rate vs validation-tuned protocol comparison tables."""
from pathlib import Path
from lpfn.benchmarking.reporting import generate_optimizer_protocol_comparison, generate_pilot_report
ROOT=Path(__file__).resolve().parents[1]
TUNED=ROOT/'results'/'selection_tuned_pilot'
FIXED=ROOT/'results'/'selection_pilot'
if __name__=='__main__':
    generate_pilot_report(TUNED, targets=('x_rotation','xz_product','noncommuting_hamiltonian'),
        depths=(1,2,3), caps=(30,60,120), seeds=(11,23,37))
    paths=generate_optimizer_protocol_comparison(FIXED,TUNED)
    for k,v in paths.items(): print(k,v)
