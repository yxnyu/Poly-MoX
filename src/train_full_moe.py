#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from feature_utils import (
    normalize_column_name,
    find_smiles_column,
    smiles_to_morgan_bits,
    compute_chemberta_embeddings,
    align_and_hstack,
    build_numeric_from_columns,
    find_column_by_key,
)
from regmix_core import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train full MoE model (ECFP+ChemBERTa) using fixed params, no CV")
    parser.add_argument("--csv", required=True, help="Training CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--hf_model", default="DeepChem/ChemBERTa-77M-MLM", help="HF model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint size")
    parser.add_argument("--cache_dir", default=".cache_features", help="Feature cache dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--params_json", required=True, help="best params JSON to use")
    parser.add_argument("--save_dir", default="runs", help="Artifacts root dir")
    args = parser.parse_args()

    # Load data and normalize columns
    df = pd.read_csv(args.csv)
    rename_map = {c: normalize_column_name(c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)
    # Drop duplicate column names (e.g. \xa0 vs space after normalize), keep last (has actual data)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    target_col = normalize_column_name(args.target)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    smiles_col = find_smiles_column(list(df.columns))
    smiles = df[smiles_col].astype(str).tolist()

    # Whether to include Initiator Percentage (%) numeric feature
    import os as _os
    WITH_INITIATOR_PCT = _os.environ.get("WITH_INITIATOR_PCT", "0") in ("1", "true", "True")
    # optional: local weighting to pull a few key points closer
    USE_LOCAL_WEIGHTING = _os.environ.get("USE_LOCAL_WEIGHTING", "0") in ("1", "true", "True")
    LOCAL_WEIGHT_MIN_ID = int(_os.environ.get("LOCAL_WEIGHT_MIN_ID", "96"))
    try:
        LOCAL_WEIGHT_FACTOR = float(_os.environ.get("LOCAL_WEIGHT_FACTOR", "3"))
    except Exception:
        LOCAL_WEIGHT_FACTOR = 3.0
    # optional: training-time constraint between λem and λex
    # Modes:
    #  - ratio:  enforce λem ≥ k-λex (existing)
    #  - affine: enforce λem ≥ (λex + b) / a  (equivalently λex ≤ a-λem − b)
    ENFORCE_EM_GE_EX_TRAIN = _os.environ.get("ENFORCE_EM_GE_EX_TRAIN", "0") in ("1", "true", "True")
    CONSTRAINT_MODE = _os.environ.get("EM_EX_CONSTRAINT_MODE", "ratio").strip().lower()
    try:
        RATIO_K = float(_os.environ.get("RATIO_K", "1.0"))
    except Exception:
        RATIO_K = 1.0
    try:
        AFFINE_A = float(_os.environ.get("AFFINE_A", "0.844"))
    except Exception:
        AFFINE_A = 0.844
    try:
        AFFINE_B = float(_os.environ.get("AFFINE_B", "12.0"))
    except Exception:
        AFFINE_B = 12.0

    # Build features
    X_morgan, idx_morgan = smiles_to_morgan_bits(smiles, radius=args.radius, n_bits=args.n_bits, cache_dir=args.cache_dir)
    X_chem, idx_chem = compute_chemberta_embeddings(
        smiles,
        hf_model_name=args.hf_model,
        batch_size=32,
        max_length=256,
        device=args.device,
        pooling="mean",
        cache_dir=args.cache_dir,
    )
    feats = [(X_morgan, idx_morgan), (X_chem, idx_chem)]
    pct_colname = None
    if WITH_INITIATOR_PCT:
        cand = find_column_by_key(list(df.columns), "Initiator Percentage (%)")
        if cand is None:
            raise ValueError("Initiator Percentage (%) feature enabled but column not found in CSV")
        X_num, idx_num = build_numeric_from_columns(df, [cand])
        feats.append((X_num, idx_num))
        pct_colname = normalize_column_name(cand)
    X_all, common_idx = align_and_hstack(feats)
    n_morgan = X_morgan.shape[1]
    n_chem = X_chem.shape[1]
    X_ecfp = X_all[:, :n_morgan]
    X_chem_only = X_all[:, n_morgan:n_morgan + n_chem]
    X_num_aligned = None
    if WITH_INITIATOR_PCT:
        X_num_aligned = X_all[:, n_morgan + n_chem:]
    y = df[target_col].values[common_idx].astype(float)

    # if training λem and constraint is enabled: y = max(y, k * λex)
    constraint_info = {
        "enabled": False,
        "target": target_col,
        "mode": CONSTRAINT_MODE,
        "ratio_k": RATIO_K,
        "affine": {"a": AFFINE_A, "b": AFFINE_B},
        "applied_on": 0,
    }
    if ENFORCE_EM_GE_EX_TRAIN and target_col == normalize_column_name("λem"):
        lex_col = find_column_by_key(list(df.columns), "λex")
        if lex_col is not None:
            y_ex = df[lex_col].values[common_idx].astype(float)
            before = y.copy()
            if CONSTRAINT_MODE == "affine":
                # enforce y_em >= (y_ex + b) / a ; assume a>0
                denom = AFFINE_A if AFFINE_A != 0 else 1e-6
                thresh = (y_ex + AFFINE_B) / denom
                y = np.maximum(y, thresh)
                constraint_info["enabled"] = True
                constraint_info["applied_on"] = int(np.sum(before < thresh))
                print(f"[info] Training constraint (affine): λem ≥ (λex+{AFFINE_B})/{AFFINE_A} | corrected={constraint_info['applied_on']}")
            else:
                # ratio mode (default)
                y = np.maximum(y, RATIO_K * y_ex)
                constraint_info["enabled"] = True
                constraint_info["applied_on"] = int(np.sum(before < RATIO_K * y_ex))
                print(f"[info] Training constraint (ratio): λem ≥ {RATIO_K}-λex | corrected={constraint_info['applied_on']}")
        else:
            print("[warn] ENFORCE_EM_GE_EX_TRAIN=1 but no λex column in training CSV, skipping constraint")

    # Build sample weights aligned to common_idx if enabled
    sample_weight = None
    weight_info = {
        "enabled": bool(USE_LOCAL_WEIGHTING),
        "min_training_id": LOCAL_WEIGHT_MIN_ID,
        "factor": float(LOCAL_WEIGHT_FACTOR),
        "n_weighted": 0,
    }
    if USE_LOCAL_WEIGHTING:
        # Locate Training data column (normalized)
        td_col = None
        for cand in df.columns:
            if normalize_column_name(cand) == "Training data":
                td_col = cand
                break
        if td_col is None:
            print("[warn] USE_LOCAL_WEIGHTING=1 but no 'Training Data' column found, ignoring weights")
        else:
            # Robust: cast to numeric and align to common_idx
            td_series = pd.to_numeric(df[td_col], errors="coerce").fillna(-1)
            try:
                td_series = td_series.astype(int)
            except Exception:
                pass
            td_vals = td_series.values
            ids = np.array([td_vals[i] for i in common_idx])
            w = np.ones_like(y, dtype=float)
            mask = ids >= LOCAL_WEIGHT_MIN_ID
            w[mask] = LOCAL_WEIGHT_FACTOR
            sample_weight = w
            weight_info["n_weighted"] = int(mask.sum())
            print(f"[info] enabling sample weighting: id>={LOCAL_WEIGHT_MIN_ID}, factor={LOCAL_WEIGHT_FACTOR}, hits={int(mask.sum())}")

    # Load fixed params
    with open(args.params_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    # Compatible with two formats:
    # 1. new format (hyperparameter search): {"best_test_pcc": 0.8, "params": {...}, ...}
    # 2. oldformat (直receiveparams): {"xgb1_n_estimators": 500, ...}
    if "params" in loaded:
        best = loaded["params"]
        print(f"using hyperparameter search results (test PCC: {loaded.get('best_test_pcc', 'N/A')})")
    else:
        best = loaded
        print("using direct parameter file")

    def extract(prefix: str):
        out = {}
        for k, v in best.items():
            if k.startswith(prefix):
                key = k[len(prefix):]
                if key == "lr":
                    out["learning_rate"] = v
                else:
                    out[key] = v
        return out

    params1 = extract("xgb1_")
    params2 = extract("xgb2_")
    gate_alpha = float(best.get("gate_l1", 1e-3))

    # Device string for XGBoost
    try:
        import torch  # type: ignore
        gpu = (args.device != "cpu" and torch.cuda.is_available()) if args.device != "cpu" else False
    except Exception:
        gpu = False
    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and gpu)) else "cpu"

    fixed = {
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "predictor": "gpu_predictor" if device_str == "cuda" else "auto",
        "device": device_str,
        "random_state": args.seed,
    }

    # Train experts on full data
    from xgboost import XGBRegressor
    xgb1 = XGBRegressor(**{**params1, **fixed})
    xgb2 = XGBRegressor(**{**params2, **fixed})
    xgb1.fit(X_ecfp, y, sample_weight=sample_weight)
    xgb2.fit(X_chem_only, y, sample_weight=sample_weight)

    # Train gate on full predictions
    pred1 = xgb1.predict(X_ecfp)
    pred2 = xgb2.predict(X_chem_only)
    if WITH_INITIATOR_PCT and X_num_aligned is not None:
        gate_X = np.column_stack([pred1, pred2, X_num_aligned])
    else:
        gate_X = np.stack([pred1, pred2], axis=1)
    from sklearn.linear_model import ElasticNet
    gate = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, random_state=args.seed, max_iter=20000, tol=1e-5)
    try:
        gate.fit(gate_X, y, sample_weight=sample_weight)
    except TypeError:
        # Compatibleold版 sklearn not支持 sample_weight 的status
        gate.fit(gate_X, y)

    # optional training metrics (on train)
    y_hat = gate.intercept_ + gate_X @ gate.coef_.astype(float)
    tr_metrics = compute_metrics(y, y_hat)

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / f"moe_full_{args.target}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    xgb1.save_model(str(out_dir / "expert_morgan_xgb.json"))
    xgb2.save_model(str(out_dir / "expert_chem_xgb.json"))
    try:
        import joblib  # type: ignore
        joblib.dump(gate, out_dir / "gate_elasticnet.pkl")
    except Exception:
        import pickle
        with open(out_dir / "gate_elasticnet.pkl", "wb") as f:
            pickle.dump(gate, f)

    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "moe_full",
            "target": args.target,
            "device": device_str,
            "n_samples": int(len(y)),
            "train_metrics": tr_metrics,
            "features": {
                "morgan_radius": args.radius,
                "morgan_bits": args.n_bits,
                "hf_model": args.hf_model,
                "chemberta_pooling": "mean",
                "use_initiator_pct": bool(WITH_INITIATOR_PCT),
                "initiator_pct_col": pct_colname,
            },
            "constraints": constraint_info,
            "weighting": weight_info,
            "files": {
                "expert_morgan_xgb": "expert_morgan_xgb.json",
                "expert_chem_xgb": "expert_chem_xgb.json",
                "gate": "gate_elasticnet.pkl",
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ saved FULL MoE model to: {out_dir}")
    print(f"Train r={tr_metrics['pearson_r']:.4f}, R2={tr_metrics['r2']:.4f}, MAE={tr_metrics['mae']:.4f}, RMSE={tr_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
