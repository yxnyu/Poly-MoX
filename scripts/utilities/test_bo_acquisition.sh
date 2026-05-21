#!/bin/bash -e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --mem=8GB
#SBATCH --job-name=test_bo

# Smoke-test bo_acquisition.py end-to-end.
#
# Required env vars (script aborts without them):
#   POLYMOX_SIMG     — absolute path to your Singularity .sif image
#   POLYMOX_OVERLAY  — absolute path to your overlay file
# Optional:
#   POLYMOX_ROOT     — repo root on the cluster (default: $PWD)
#   SBATCH_ACCOUNT   — SLURM allocation; SLURM reads this natively

: "${POLYMOX_SIMG:?Set POLYMOX_SIMG to your Singularity image path}"
: "${POLYMOX_OVERLAY:?Set POLYMOX_OVERLAY to your overlay file path}"
WORKDIR="${POLYMOX_ROOT:-$PWD}"

singularity exec --overlay "$POLYMOX_OVERLAY" "$POLYMOX_SIMG" /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export PYTHONUSERBASE=/ext3/python_packages
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH:${WORKDIR}/src
    cd $WORKDIR

    python3 - <<'PYEOF'
import numpy as np
import pandas as pd
from bo_acquisition import (
    tanimoto_similarity, tanimoto_matrix, diversity_score,
    minmax_normalize, acquisition_ei, acquisition_ucb,
    acquisition_thompson_sample, expert_disagreement_sigma,
    composite_score, greedy_batch_select, restrict_to_phosphine,
    synth_filter_topk, select_round_batch, PHASE_WEIGHTS,
)

print('=== 1. Tanimoto similarity ===')
fp1 = np.array([1,1,0,0,1])
fp2 = np.array([1,0,1,0,1])
print(f'  Tan(fp1,fp2)={tanimoto_similarity(fp1,fp2):.4f}  (expected 0.5)')

print()
print('=== 2. Tanimoto matrix (3 fps) ===')
fps = np.array([[1,1,0,0,1],[1,0,1,0,1],[0,0,1,1,0]])
print(tanimoto_matrix(fps))

print()
print('=== 3. Diversity score (cand vs S) ===')
sel = fps[:1]
d = diversity_score(fps[1:], sel)
print(f'  d = {d}  (1 - Tanimoto)')

print()
print('=== 4. Min-max normalize ===')
print(f'  norm([1,2,3,4]) = {minmax_normalize(np.array([1,2,3,4]))}')

print()
print('=== 5. Acquisition functions ===')
mu = np.array([1.0, 2.0, 3.0])
sigma = np.array([0.5, 0.1, 0.3])
print(f'  EI(y_best=2):  {acquisition_ei(mu, sigma, 2.0)}')
print(f'  UCB(beta=2):   {acquisition_ucb(mu, sigma, beta=2)}')
print(f'  TS(seed=42):   {acquisition_thompson_sample(mu, sigma, seed=42)}')

print()
print('=== 6. Expert disagreement σ ===')
pm = np.array([400.0, 500.0, 600.0])
pc = np.array([420.0, 480.0, 580.0])
print(f'  σ proxy = {expert_disagreement_sigma(pm, pc)}')

print()
print('=== 7. Composite score ===')
a = np.array([0.1, 0.5, 0.9])
d = np.array([0.9, 0.5, 0.1])
print(f'  s(w_r=0.10) = {composite_score(a, d, 0.10)}')
print(f'  s(w_r=0.25) = {composite_score(a, d, 0.25)}')
print(f'  PHASE_WEIGHTS = {PHASE_WEIGHTS}')

print()
print('=== 8. Greedy batch select (5 candidates → B=3) ===')
rng = np.random.default_rng(42)
fps5 = (rng.random((5, 2048)) > 0.95).astype(np.int32)
acq5 = np.array([0.1, 0.8, 0.3, 0.9, 0.5])
ids5 = [101, 102, 103, 104, 105]
batch_w010 = greedy_batch_select(acq5, fps5, ids5, batch_size=3, w_r=0.10)
batch_w050 = greedy_batch_select(acq5, fps5, ids5, batch_size=3, w_r=0.50)
print(f'  w_r=0.10 → batch={batch_w010}  (acq-driven)')
print(f'  w_r=0.50 → batch={batch_w050}  (more diversity-aware)')

print()
print('=== 9. Subspace restriction (phosphine filter) ===')
test_df = pd.DataFrame({
    'Training data': [1, 2, 3, 4],
    'Initiator ': ['CCN(CC)CC', 'P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3', '[H]O[H]', 'CC(C)(C)P(C(C)(C)C)C(C)(C)C'],
})
filtered = restrict_to_phosphine(test_df)
print(f'  original {len(test_df)} → phosphine {len(filtered)}: IDs={filtered[\"Training data\"].tolist()}  (expect 2,4)')

print()
print('=== 10. Synthesizability top-K filter ===')
ranked = [10, 20, 30, 40, 50]
feasible = [10, 30, 50, 70]
print(f'  topK=4, feasible=[10,30,50,70] → {synth_filter_topk(ranked, feasible, K=4, batch_size=3)}  (expect [10,30,50])')

print()
print('=== 11. End-to-end select_round_batch (R1 real data) ===')
pool = pd.read_csv('data/merged_final_classified.csv')
pool = pool[pool['Round']=='R1'].reset_index(drop=True)
print(f'  candidate pool size: {len(pool)} (R1 batch from merged_final_classified.csv)')

mu = np.linspace(0, 1, len(pool))  # fake predictions
ts_batch = select_round_batch(
    pool_df=pool, pred_mu=mu, round_idx=1, batch_size=5,
    acquisition='thompson', cache_dir='.cache_features', seed=42,
)
print(f'  TS, R1, B=5 → selected IDs: {ts_batch}')

ucb_batch = select_round_batch(
    pool_df=pool, pred_mu=mu, pred_sigma=np.full(len(pool), 0.1),
    round_idx=2, batch_size=5, acquisition='ucb', cache_dir='.cache_features',
)
print(f'  UCB, R2, B=5 → selected IDs: {ucb_batch}')

print()
print('=== all tests passed ✓ ===')
PYEOF
"
