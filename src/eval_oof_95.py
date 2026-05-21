#!/usr/bin/env python3
"""
Use saved best_params.json to do 5-fold OOF evaluation on 95 training_original samples, 
Verify the cv_metrics recorded in meta.json can be reproduced. 
Usage:
  python eval_oof_95.py --ckpt checkpoints/lem_sep16 --csv data/training_original_95.csv
"""
import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
import xgboost as xgb

from feature_utils import (
    normalize_column_name,
    find_smiles_column,
    smiles_to_morgan_bits,
    compute_chemberta_embeddings,
    align_and_hstack,
    build_numeric_from_columns,
    find_column_by_key,
)


def pearson_r(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint directory (contains meta.json + best_params.json)")
    parser.add_argument("--csv", required=True, help="Training data CSV (95 samples)")
    parser.add_argument("--cache_dir", default=".cache_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--device", default=None, help="override the device setting in meta.json")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    params = json.loads((ckpt / "best_params.json").read_text())

    target_col_raw = meta["target"]
    feats = meta["features"]
    radius   = int(feats.get("morgan_radius", 2))
    n_bits   = int(feats.get("morgan_bits", 2048))
    hf_model = feats.get("hf_model", "DeepChem/ChemBERTa-77M-MLM")
    pooling  = feats.get("chemberta_pooling", "mean")
    device   = args.device if args.device else meta.get("device", "cpu")

    print(f"Target : {target_col_raw}")
    print(f"Features: morgan r={radius} b={n_bits}  |  {hf_model}")
    print(f"Params  : {params}")

    # ── Loaddata ──────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    rename_map = {c: normalize_column_name(c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    target_col = normalize_column_name(target_col_raw)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = df[target_col].astype(float).values
    smiles_col = find_smiles_column(list(df.columns))
    smiles = df[smiles_col].astype(str).tolist()
    print(f"n_samples: {len(df)}")

    # ── featuresextract ──────────────────────────────────────────────
    X_morgan, idx_m = smiles_to_morgan_bits(smiles, radius=radius, n_bits=n_bits,
                                             cache_dir=args.cache_dir)
    X_chem, idx_c = compute_chemberta_embeddings(smiles, hf_model_name=hf_model,
                                                  batch_size=32, max_length=256,
                                                  device=device, pooling=pooling,
                                                  cache_dir=args.cache_dir)
    pct_col = find_column_by_key(list(df.columns), "Initiator Percentage (%)")
    import os
    with_pct = os.environ.get("WITH_INITIATOR_PCT", "0") in ("1", "true", "True")
    X_full, idx_shared = align_and_hstack([(X_morgan, idx_m), (X_chem, idx_c)])
    n_morgan = X_morgan.shape[1]
    X_morgan_aligned = X_full[:, :n_morgan]
    X_chem_aligned   = X_full[:, n_morgan:]

    X_num_aligned = None
    if with_pct and pct_col:
        X_num, idx_n = build_numeric_from_columns(df, [pct_col])
        # Move numeric alignto idx_shared sequential
        pos = {idx: i for i, idx in enumerate(idx_n)}
        X_num_aligned = np.stack([X_num[pos[i]] for i in idx_shared], axis=0)
        feat_dim_total = X_full.shape[1] + X_num_aligned.shape[1]
    else:
        feat_dim_total = X_full.shape[1]

    y_aligned = y[idx_shared]
    print(f"n_aligned: {len(idx_shared)}  feature_dim: {feat_dim_total}")

    # ── Parse hyperparameters (identical to extract() in moe_optuna.py)─────
    def extract(prefix):
        out = {}
        for k, v in params.items():
            if k.startswith(prefix):
                key = k[len(prefix):]
                out["learning_rate" if key == "lr" else key] = v
        return out

    p1 = extract("xgb1_")
    p2 = extract("xgb2_")
    gate_alpha = float(params.get("gate_l1", 1e-3))

    import torch as _torch
    gpu = (_torch.cuda.is_available() and device != "cpu")
    device_str = "cuda" if (device == "cuda" or (device == "auto" and gpu)) else "cpu"
    fixed = {
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "predictor": "gpu_predictor" if device_str == "cuda" else "auto",
        "device": device_str,
        "random_state": args.seed,
    }
    p1 = {**p1, **fixed}
    p2 = {**p2, **fixed}

    # ── 5-fold OOF (identical to original moe_optuna.py)────────────
    # XGB uses only Morgan / ChemBERTa (no numeric); numeric is added separately to gate_X
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_morgan = np.zeros(len(idx_shared))
    oof_chem   = np.zeros(len(idx_shared))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_morgan_aligned)):
        m1 = xgb.XGBRegressor(**p1)
        m1.fit(X_morgan_aligned[tr_idx], y_aligned[tr_idx])
        oof_morgan[val_idx] = m1.predict(X_morgan_aligned[val_idx])

        m2 = xgb.XGBRegressor(**p2)
        m2.fit(X_chem_aligned[tr_idx], y_aligned[tr_idx])
        oof_chem[val_idx] = m2.predict(X_chem_aligned[val_idx])
        print(f"  Fold {fold+1}: morgan_r={pearson_r(y_aligned[val_idx], oof_morgan[val_idx]):.4f}  chem_r={pearson_r(y_aligned[val_idx], oof_chem[val_idx]):.4f}")

    # gate: ifhave numeric features, 加to gate_X (withoriginalcodeconsistent)
    if X_num_aligned is not None:
        gate_X = np.column_stack([oof_morgan, oof_chem, X_num_aligned])
    else:
        gate_X = np.column_stack([oof_morgan, oof_chem])
    gate = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, fit_intercept=True,
                      random_state=args.seed, max_iter=20000, tol=1e-5)
    gate.fit(gate_X, y_aligned)
    oof_pred = gate.intercept_ + gate_X @ gate.coef_.astype(float)

    # ── summary指标 ──────────────────────────────────────────────
    r   = pearson_r(y_aligned, oof_pred)
    ss_res = np.sum((y_aligned - oof_pred) ** 2)
    ss_tot = np.sum((y_aligned - y_aligned.mean()) ** 2)
    r2  = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_aligned - oof_pred)))
    rms = rmse(y_aligned, oof_pred)

    print()
    print("---- Reproduced 5-fold OOF metrics ----")
    print(f"r={r:.4f}, R2={r2:.4f}, MAE={mae:.4f}, RMSE={rms:.4f}")
    print()
    print("---- meta.json original record ----")
    cv = meta.get("cv_metrics")
    if cv:
        print(f"r={cv['pearson_r']:.4f}, R2={cv['r2']:.4f}, "
              f"MAE={cv['mae']:.4f}, RMSE={cv['rmse']:.4f}")
    else:
        tm = meta.get("train_metrics")
        if tm:
            print(f"(No cv_metrics; train_metrics: r={tm['pearson_r']:.4f}, R2={tm['r2']:.4f}, "
                  f"MAE={tm['mae']:.4f}, RMSE={tm['rmse']:.4f})")
        else:
            print("(meta.json has neither cv_metrics nor train_metrics)")


if __name__ == "__main__":
    main()
