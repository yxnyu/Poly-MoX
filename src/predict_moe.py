#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

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


def load_gate(path: Path):
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict QY using saved MoE artifacts")
    parser.add_argument("--csv", required=True, help="CSV file with molecules to screen")
    parser.add_argument("--artifacts_dir", required=True, help="Directory with saved models (expert_xgb + gate + meta.json)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <input>_pred.csv)")
    parser.add_argument("--cache_dir", default=".cache_features", help="Feature cache directory")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    meta_path = art / "meta.json"
    xgb1_path = art / "expert_morgan_xgb.json"
    xgb2_path = art / "expert_chem_xgb.json"
    gate_path = art / "gate_elasticnet.pkl"
    assert meta_path.exists(), f"meta.json not found in {art}"
    assert xgb1_path.exists(), f"expert_morgan_xgb.json not found in {art}"
    assert xgb2_path.exists(), f"expert_chem_xgb.json not found in {art}"
    assert gate_path.exists(), f"gate_elasticnet.pkl not found in {art}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feats = meta.get("features", {})
    radius = int(feats.get("morgan_radius", 2))
    n_bits = int(feats.get("morgan_bits", 2048))
    hf_model = str(feats.get("hf_model", "DeepChem/ChemBERTa-77M-MLM"))
    pooling = str(feats.get("chemberta_pooling", "mean"))

    # Read input CSV
    df = pd.read_csv(args.csv)
    rename_map = {c: normalize_column_name(c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

    smiles_col = find_smiles_column(list(df.columns))
    smiles = df[smiles_col].astype(str).tolist()

    # Build features with same config as training
    X_morgan, idx_morgan = smiles_to_morgan_bits(smiles, radius=radius, n_bits=n_bits, cache_dir=args.cache_dir)
    X_chem, idx_chem = compute_chemberta_embeddings(
        smiles,
        hf_model_name=hf_model,
        batch_size=32,
        max_length=256,
        device="auto",
        pooling=pooling,
        cache_dir=args.cache_dir,
    )
    # optionally include numeric Initiator Percentage (%) if training meta says so
    feats = [(X_morgan, idx_morgan), (X_chem, idx_chem)]
    X_num_aligned = None
    pct_used = meta.get("features", {}).get("use_initiator_pct", False)
    pct_col = meta.get("features", {}).get("initiator_pct_col")
    if pct_used:
        cand = None
        if pct_col:
            # meta already stores normalized name; try match by key (robust to slight differences)
            cand = find_column_by_key(list(df.columns), pct_col) or find_column_by_key(list(df.columns), "Initiator Percentage (%)")
        else:
            cand = find_column_by_key(list(df.columns), "Initiator Percentage (%)")
        if cand is None:
            raise ValueError("该ModelatTrainingwhen使used Initiator Percentage (%), 但PredictCSVinnot foundto应列")
        X_num, idx_num = build_numeric_from_columns(df, [cand])
        feats.append((X_num, idx_num))

    X_all, common_idx = align_and_hstack(feats)
    n_morgan = X_morgan.shape[1]
    n_chem = X_chem.shape[1]
    X_ecfp = X_all[:, :n_morgan]
    X_chem_only = X_all[:, n_morgan:n_morgan + n_chem]
    if pct_used:
        X_num_aligned = X_all[:, n_morgan + n_chem:]

    # Load models
    from xgboost import XGBRegressor
    xgb1 = XGBRegressor()
    xgb1.load_model(str(xgb1_path))
    xgb2 = XGBRegressor()
    xgb2.load_model(str(xgb2_path))
    gate = load_gate(gate_path)

    # Predict with two experts then gate
    pred1 = xgb1.predict(X_ecfp)
    pred2 = xgb2.predict(X_chem_only)
    if pct_used and X_num_aligned is not None:
        gate_X = np.column_stack([pred1, pred2, X_num_aligned])
    else:
        gate_X = np.stack([pred1, pred2], axis=1)
    coef = np.asarray(getattr(gate, "coef_", [0.5, 0.5]), dtype=float).reshape(1, -1)
    intercept = float(getattr(gate, "intercept_", 0.0))
    y_pred = intercept + (gate_X @ coef.T).ravel()

    # Stitch back to full dataframe
    out_df = df.copy()
    pred_col = f"{meta.get('target', 'QY')}_pred"
    out_df[pred_col] = np.nan
    # common_idx are row indices of the original df that are valid for both features
    for pos, idx in enumerate(common_idx):
        out_df.at[idx, pred_col] = float(y_pred[pos])

    out_path = args.output
    if not out_path:
        inp = Path(args.csv)
        out_path = str(inp.with_name(inp.stem + "_pred.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"✅ saved predictions to: {out_path}")
    valid_n = int(np.isfinite(out_df[pred_col]).sum())
    print(f"predicted {valid_n} / {len(out_df)} rows (valid for both features) -> column: {pred_col}")


if __name__ == "__main__":
    main()
