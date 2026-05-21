#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from feature_utils import (
    normalize_column_name, find_smiles_column,
    smiles_to_morgan_bits, compute_chemberta_embeddings, align_and_hstack,
    build_numeric_from_columns, find_column_by_key
)
from regmix_core import optuna_moe, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="MoE (ECFP+ChemBERTa) Optuna hyperparameter tuning")
    parser.add_argument("--csv", required=True, help="CSV filepath")
    parser.add_argument("--target", default="QY", help="Target column name")
    parser.add_argument("--hf_model", default="DeepChem/ChemBERTa-77M-MLM", help="HuggingFace Modelname")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius")
    parser.add_argument("--n_bits", type=int, default=2048, help="fingerprint size")
    parser.add_argument("--optuna_trials", type=int, default=600, help="Number of Optuna trials")
    parser.add_argument("--cache_dir", default=".cache_features", help="Feature cache directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_best", action="store_true", help="Save best modelwithparams")
    # Optional: include Initiator Percentage (%) numeric column
    # For backward compatibility with existing scripts, use env var or simple probing instead of adding required params
    import os as _os
    WITH_INITIATOR_PCT = _os.environ.get("WITH_INITIATOR_PCT", "0") in ("1", "true", "True")
    parser.add_argument("--save_dir", default="runs", help="Save directory (default: runs)")
    parser.add_argument("--params_json", default=None, help="Load best params from JSON, skipping Optuna search")
    parser.add_argument("--storage", default=None, help="Optuna SQLite storage (sqlite:///path.db) to support auto-resume")
    parser.add_argument("--study_name", default=None, help="Optuna study name (with storage 配合使use)")
    parser.add_argument("--round_filter", default=None, help="Optional: filter rows by Round column value (e.g. 'training_original' for the 95-sample subset)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.round_filter and 'Round' in df.columns:
        df = df[df['Round'] == args.round_filter].reset_index(drop=True)
    rename_map = {c: normalize_column_name(c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    target_col = normalize_column_name(args.target)
    if target_col not in df.columns:
        raise ValueError(f"not foundTarget column: {args.target}")

    # Build Morgan with ChemBERTa features
    smiles_col = find_smiles_column(list(df.columns))
    smiles_series = df[smiles_col].astype(str).tolist()

    X_morgan, idx_morgan = smiles_to_morgan_bits(
        smiles_series, radius=args.radius, n_bits=args.n_bits, cache_dir=args.cache_dir
    )
    X_chem, idx_chem = compute_chemberta_embeddings(
        smiles_series,
        hf_model_name=args.hf_model,
        batch_size=32,
        max_length=256,
        device=args.device,
        pooling="mean",
        cache_dir=args.cache_dir,
    )
    X_all, common_idx = align_and_hstack([(X_morgan, idx_morgan), (X_chem, idx_chem)])
    features = [(X_morgan, idx_morgan), (X_chem, idx_chem)]
    pct_colname = None
    if WITH_INITIATOR_PCT:
        cand = find_column_by_key(list(df.columns), "Initiator Percentage (%)")
        if cand is None:
            raise ValueError("已启use Initiator Percentage (%) features, 但atCSVinnot found该列")
        X_num, idx_num = build_numeric_from_columns(df, [cand])
        features.append((X_num, idx_num))
        pct_colname = normalize_column_name(cand)
    X_all, common_idx = align_and_hstack(features)
    n_morgan = X_morgan.shape[1]
    n_chem = X_chem.shape[1]
    X_ecfp = X_all[:, :n_morgan]
    X_chem_only = X_all[:, n_morgan:n_morgan + n_chem]
    X_num_aligned = None
    if WITH_INITIATOR_PCT:
        X_num_aligned = X_all[:, n_morgan + n_chem:]
    y = df[target_col].values[common_idx].astype(float)
    # Training-time physics constraint for λem (defaults OFF here; tune_chain.sh sets the
    # env var ENFORCE_EM_GE_EX_TRAIN=1 by default so a clean repo run gets the constraint).
    # Mechanism: clamp the training labels y_em so they satisfy the chosen physical relation
    # with y_ex BEFORE Optuna sees them.  This biases the fitted model toward physical predictions
    # without requiring a custom XGBoost loss.
    import os as _os
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
                denom = AFFINE_A if AFFINE_A != 0 else 1e-6
                thresh = (y_ex + AFFINE_B) / denom
                y = np.maximum(y, thresh)
                constraint_info["enabled"] = True
                constraint_info["applied_on"] = int(np.sum(before < thresh))
                print(f"[info] Training constraint (affine): λem ≥ (λex+{AFFINE_B})/{AFFINE_A} | corrected={constraint_info['applied_on']}")
            else:
                y = np.maximum(y, RATIO_K * y_ex)
                constraint_info["enabled"] = True
                constraint_info["applied_on"] = int(np.sum(before < RATIO_K * y_ex))
                print(f"[info] Training constraint (ratio): λem ≥ {RATIO_K}·λex | corrected={constraint_info['applied_on']}")
        else:
            print("[warn] ENFORCE_EM_GE_EX_TRAIN=1 but no λex column in training CSV, skipping constraint")

    # chooseparams:固定paramsor Optuna Search
    import json
    import os as _os
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            loaded_params = json.load(f)
        res = {"best_params": loaded_params, "n_samples": len(y), "best_r2": None}
        search_mode = "固定params"
    else:
        res = optuna_moe(X_ecfp, X_chem_only, y, n_trials=args.optuna_trials,
                          random_state=args.seed,
                          storage=args.storage, study_name=args.study_name,
                          target=args.target)
        search_mode = "Optuna search"
    print("==== 🧠 MoE (ECFP+ChemBERTa) ====")
    print(f"模式: {search_mode}")
    if res.get("best_r2") is not None:
        print(f"Best R²: {res['best_r2']:.6f}")
    print(f"n_samples: {res['n_samples']}")
    print("best params:")
    print(json.dumps(res["best_params"], ensure_ascii=False, indent=2))

    # === 基于best params的 5-fold OOF evaluation: r / R2 / MAE / RMSE ===
    best = dict(res["best_params"])  # 拷贝to便processbefore缀
    def extract(prefix: str) -> dict:
        out = {}
        for k, v in list(best.items()):
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
    # devicechoose
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
    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold
    from sklearn.linear_model import ElasticNet
    # Local weighting config
    USE_LOCAL_WEIGHTING = _os.environ.get("USE_LOCAL_WEIGHTING", "0") in ("1", "true", "True")
    LOCAL_WEIGHT_MIN_ID = int(_os.environ.get("LOCAL_WEIGHT_MIN_ID", "96"))
    try:
        LOCAL_WEIGHT_FACTOR = float(_os.environ.get("LOCAL_WEIGHT_FACTOR", "3"))
    except Exception:
        LOCAL_WEIGHT_FACTOR = 3.0
    sample_weight = None
    weight_info = {
        "enabled": bool(USE_LOCAL_WEIGHTING),
        "min_training_id": LOCAL_WEIGHT_MIN_ID,
        "factor": float(LOCAL_WEIGHT_FACTOR),
        "n_weighted": 0,
    }
    if USE_LOCAL_WEIGHTING:
        # Find Training data column in df (already normalized names)
        td_col = None
        for cand in df.columns:
            if normalize_column_name(cand) == "Training data":
                td_col = cand
                break
        if td_col is None:
            print("[warn] USE_LOCAL_WEIGHTING=1 but no 'Training Data' column found, ignoring weights")
        else:
            # Robust: cast to numeric ids aligned to common_idx
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
    oof_ecfp = np.zeros_like(y, dtype=float)
    oof_chem = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr, te in kf.split(X_ecfp):
        xgb1 = XGBRegressor(**{**params1, **fixed})
        if sample_weight is not None:
            xgb1.fit(X_ecfp[tr], y[tr], sample_weight=sample_weight[tr])
        else:
            xgb1.fit(X_ecfp[tr], y[tr])
        oof_ecfp[te] = xgb1.predict(X_ecfp[te])
        xgb2 = XGBRegressor(**{**params2, **fixed})
        if sample_weight is not None:
            xgb2.fit(X_chem_only[tr], y[tr], sample_weight=sample_weight[tr])
        else:
            xgb2.fit(X_chem_only[tr], y[tr])
        oof_chem[te] = xgb2.predict(X_chem_only[te])
    if WITH_INITIATOR_PCT and X_num_aligned is not None:
        gate_X = np.column_stack([oof_ecfp, oof_chem, X_num_aligned])
    else:
        gate_X = np.stack([oof_ecfp, oof_chem], axis=1)
    gate = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, random_state=args.seed, max_iter=20000, tol=1e-5)
    try:
        gate.fit(gate_X, y, sample_weight=sample_weight)
    except TypeError:
        gate.fit(gate_X, y)
    oof_final = gate.intercept_ + gate_X @ gate.coef_.astype(float)
    metrics = compute_metrics(y, oof_final)
    print("---- 5-fold OOF metrics using best params ----")
    print(f"r={metrics['pearson_r']:.4f}, R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    # optional: Save best modelwithparams
    if args.save_best:
        from pathlib import Path
        from datetime import datetime
        try:
            import torch  # type: ignore
            gpu = (args.device != "cpu" and torch.cuda.is_available()) if args.device != "cpu" else False
        except Exception:
            gpu = False
        device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and gpu)) else "cpu"
        # TrainingFullexpert
        xgb1 = XGBRegressor(**{**params1, **fixed})
        xgb2 = XGBRegressor(**{**params2, **fixed})
        if sample_weight is not None:
            xgb1.fit(X_ecfp, y, sample_weight=sample_weight)
        else:
            xgb1.fit(X_ecfp, y)
        if sample_weight is not None:
            xgb2.fit(X_chem_only, y, sample_weight=sample_weight)
        else:
            xgb2.fit(X_chem_only, y)
        pred1 = xgb1.predict(X_ecfp)
        pred2 = xgb2.predict(X_chem_only)
        if WITH_INITIATOR_PCT and X_num_aligned is not None:
            gate_X_full = np.column_stack([pred1, pred2, X_num_aligned])
        else:
            gate_X_full = np.stack([pred1, pred2], axis=1)
        gate_full = ElasticNet(alpha=gate_alpha, l1_ratio=0.9, random_state=args.seed, max_iter=20000, tol=1e-5)
        try:
            gate_full.fit(gate_X_full, y, sample_weight=sample_weight)
        except TypeError:
            gate_full.fit(gate_X_full, y)
        # Save产物
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.save_dir) / f"moe_optuna_{args.target}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        xgb1.save_model(str(out_dir / "expert_morgan_xgb.json"))
        xgb2.save_model(str(out_dir / "expert_chem_xgb.json"))
        # Gate
        try:
            import joblib  # type: ignore
            joblib.dump(gate_full, out_dir / "gate_elasticnet.pkl")
        except Exception:
            import pickle
            with open(out_dir / "gate_elasticnet.pkl", "wb") as f:
                pickle.dump(gate_full, f)
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(res["best_params"], f, ensure_ascii=False, indent=2)
        # Saveevaluation metric
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "pearson_r": metrics["pearson_r"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }, f, ensure_ascii=False, indent=2)
        meta = {
            "mode": "moe",
            "target": args.target,
            "device": device_str,
            "best_cv_r2": (res["best_r2"] if res.get("best_r2") is not None else metrics["r2"]),
            "cv_metrics": metrics,
            "n_samples": res.get("n_samples", len(y)),
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
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"💾 saved best MoE model to: {out_dir}")


if __name__ == "__main__":
    main()
