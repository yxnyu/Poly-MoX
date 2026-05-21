#!/bin/bash -e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
#SBATCH --job-name=repro_tuned

# Reproduce the 5-fold OOF metrics for the four 95-sample studies using the
# best_params snapshots in best_params/optuna/.  Loads the 140-row
# merged_final_classified.csv and filters to Round=='training_original'.
#
# Required env vars (script aborts without them):
#   POLYMOX_SIMG     — absolute path to your Singularity .sif image
#   POLYMOX_OVERLAY  — absolute path to your overlay file
# Optional:
#   POLYMOX_ROOT     — repo root on the cluster (default: $PWD)
#   SBATCH_ACCOUNT   — SLURM allocation; SLURM reads this natively, no explicit handling needed
#
# Submit:
#   export POLYMOX_SIMG=/path/to/image.sif
#   export POLYMOX_OVERLAY=/path/to/overlay.ext3:ro
#   export SBATCH_ACCOUNT=your_alloc       # or: sbatch --account=foo scripts/reproduce/...
#   sbatch scripts/reproduce/repro_tuned_params.sh

: "${POLYMOX_SIMG:?Set POLYMOX_SIMG to your Singularity image path}"
: "${POLYMOX_OVERLAY:?Set POLYMOX_OVERLAY to your overlay file path}"
WORKDIR="${POLYMOX_ROOT:-$PWD}"

singularity exec --nv --overlay "$POLYMOX_OVERLAY" "$POLYMOX_SIMG" /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export HF_HOME=/ext3/hf_cache
    export PYTHONUSERBASE=/ext3/python_packages
    export PATH=\$PYTHONUSERBASE/bin:\$PATH
    export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH:${WORKDIR}/src
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export WORKDIR=${WORKDIR}
    cd ${WORKDIR}

    python3 - <<'PYEOF'
import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import xgboost as xgb
from feature_utils import (normalize_column_name, find_smiles_column,
                            smiles_to_morgan_bits, compute_chemberta_embeddings,
                            align_and_hstack, build_numeric_from_columns, find_column_by_key)

CV = Path(os.environ.get('WORKDIR', '.')).resolve()
CSV_95 = 'data/merged_final_classified.csv'  # filtered to Round=='training_original' below

JOBS = [
    ('95_lex', CSV_95, 'λex', '95_lex_best.json', 123),
    ('95_lem', CSV_95, 'λem', '95_lem_best.json', 42),
    ('95_qy',  CSV_95, 'QY',  '95_qy_best.json',  42),
]

print(f'{\"tag\":<18}{\"orig_R²\":<12}{\"orig_PCC\":<12}{\"nested_R²\":<12}{\"nested_PCC\":<12}')
print('-'*72)

for tag, csv_path, target, json_name, SEED in JOBS:
    params_path = CV / 'best_params' / 'optuna' / json_name
    if not params_path.exists():
        print(f'{tag}: missing {json_name}'); continue
    params = json.loads(params_path.read_text())

    df = pd.read_csv(CV / csv_path)
    if 'Round' in df.columns:
        df = df[df['Round'] == 'training_original'].reset_index(drop=True)
    df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    target_col = normalize_column_name(target)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = df[target_col].astype(float).values

    smiles_col = find_smiles_column(list(df.columns))
    smiles = df[smiles_col].astype(str).tolist()

    Xm, idx_m = smiles_to_morgan_bits(smiles, radius=2, n_bits=2048, cache_dir='.cache_features')
    Xc, idx_c = compute_chemberta_embeddings(smiles, hf_model_name='DeepChem/ChemBERTa-77M-MLM',
                                              batch_size=32, max_length=256, device='cuda',
                                              pooling='mean', cache_dir='.cache_features')

    X_full, idx_shared = align_and_hstack([(Xm, idx_m), (Xc, idx_c)])
    n_morgan = Xm.shape[1]
    Xm_a = X_full[:, :n_morgan]
    Xc_a = X_full[:, n_morgan:]

    # WITH_INITIATOR_PCT=1 by design for the 95-sample studies
    pct_col = find_column_by_key(list(df.columns), 'Initiator Percentage (%)')
    if pct_col:
        Xnum, idx_n = build_numeric_from_columns(df, [pct_col])
        pos = {idx: i for i, idx in enumerate(idx_n)}
        Xnum_a = np.stack([Xnum[pos[i]] for i in idx_shared], axis=0)
    else:
        Xnum_a = None

    y_aligned = y[idx_shared]

    def extract(prefix):
        out = {}
        for k, v in params.items():
            if k.startswith(prefix):
                key = k[len(prefix):]
                out['learning_rate' if key=='lr' else key] = v
        return out
    p1 = extract('xgb1_')
    p2 = extract('xgb2_')
    gate_alpha = float(params.get('gate_l1', 1e-3))
    fixed = {'reg_lambda': 1.0, 'objective': 'reg:squarederror',
             'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED}
    p1 = {**p1, **fixed}; p2 = {**p2, **fixed}

    # 5-fold CV: compute both the original (full-OOF gate) and strict nested-CV metrics
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_m = np.zeros(len(idx_shared))
    oof_c = np.zeros(len(idx_shared))
    nested_pred = np.zeros(len(idx_shared))

    for tr, te in kf.split(Xm_a):
        m1 = xgb.XGBRegressor(**p1); m1.fit(Xm_a[tr], y_aligned[tr])
        pred_m_tr = m1.predict(Xm_a[tr]); oof_m[te] = m1.predict(Xm_a[te])
        m2 = xgb.XGBRegressor(**p2); m2.fit(Xc_a[tr], y_aligned[tr])
        pred_c_tr = m2.predict(Xc_a[tr]); oof_c[te] = m2.predict(Xc_a[te])

        # Strict nested CV: train gate only on the training fold
        if Xnum_a is not None:
            gX_tr = np.column_stack([pred_m_tr, pred_c_tr, Xnum_a[tr]])
            gX_te = np.column_stack([oof_m[te], oof_c[te], Xnum_a[te]])
        else:
            gX_tr = np.column_stack([pred_m_tr, pred_c_tr])
            gX_te = np.column_stack([oof_m[te], oof_c[te]])
        gate_fold = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, fit_intercept=True,
                                random_state=SEED, max_iter=20000, tol=1e-5)
        gate_fold.fit(gX_tr, y_aligned[tr])
        nested_pred[te] = gate_fold.intercept_ + gX_te @ gate_fold.coef_.astype(float)

    # Original pipeline: gate trained on full OOF and evaluated on the same OOF
    # (matches what moe_optuna.py optimised during the search).
    if Xnum_a is not None:
        gate_X = np.column_stack([oof_m, oof_c, Xnum_a])
    else:
        gate_X = np.column_stack([oof_m, oof_c])
    gate = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, fit_intercept=True,
                      random_state=SEED, max_iter=20000, tol=1e-5)
    gate.fit(gate_X, y_aligned)
    oof_final = gate.intercept_ + gate_X @ gate.coef_.astype(float)

    r2_orig    = r2_score(y_aligned, oof_final)
    pcc_orig   = float(np.corrcoef(y_aligned, oof_final)[0,1])
    r2_nested  = r2_score(y_aligned, nested_pred)
    pcc_nested = float(np.corrcoef(y_aligned, nested_pred)[0,1])
    print(f'{tag:<18}{r2_orig:<12.4f}{pcc_orig:<12.4f}{r2_nested:<12.4f}{pcc_nested:<12.4f}')

PYEOF
"
