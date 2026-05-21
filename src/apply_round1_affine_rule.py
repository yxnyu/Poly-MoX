#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

from feature_utils import normalize_column_name, find_smiles_column


def pick_join_key(df1: pd.DataFrame, df2: pd.DataFrame):
    c1 = {normalize_column_name(c): c for c in df1.columns}
    c2 = {normalize_column_name(c): c for c in df2.columns}
    if "Training data" in c1 and "Training data" in c2:
        return c1["Training data"], c2["Training data"], "Training data"
    k1 = find_smiles_column(list(df1.columns))
    k2 = find_smiles_column(list(df2.columns))
    return k1, k2, normalize_column_name(k1)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Round1 rule: use λem to infer λex via inverse equation. "
            "λem = A*λex + B → λex_calc = (λem - B) / A. "
            "if λem < λex_pred (violate), replace λex with λex_calc; "
            "else average (λex_calc + λex_pred)/2. Keep λem unchanged."
        )
    )
    ap.add_argument("--lex_csv", required=True, help="path to jianyu_prediction_pred_lex.csv")
    ap.add_argument("--lem_csv", required=True, help="path to jianyu_prediction_pred_lem.csv")
    ap.add_argument("--A", type=float, default=0.9327, help="Coefficient A in λem = A*λex + B")
    ap.add_argument("--B", type=float, default=121.8606, help="intercept B in λem = A*λex + B")
    ap.add_argument("--output", required=False, help="Output CSV path (merged with shrink columns)")
    args = ap.parse_args()

    df_ex = pd.read_csv(args.lex_csv)
    df_em = pd.read_csv(args.lem_csv)

    # Normalize column names
    df_ex.rename(columns={c: normalize_column_name(c) for c in df_ex.columns}, inplace=True)
    df_em.rename(columns={c: normalize_column_name(c) for c in df_em.columns}, inplace=True)

    # Locate prediction columns
    ex_col = next((c for c in df_ex.columns if normalize_column_name(c) in ["λex_pred", "lex_pred"]), None)
    em_col = next((c for c in df_em.columns if normalize_column_name(c) in ["λem_pred", "lem_pred"]), None)
    if ex_col is None or em_col is None:
        raise SystemExit("not found λex_pred/lex_pred or λem_pred/lem_pred 列")

    # Join rows
    k1, k2, k_norm = pick_join_key(df_ex, df_em)
    merged = df_ex[[k1, ex_col]].merge(df_em[[k2, em_col]], left_on=k1, right_on=k2, how="inner")
    if merged.empty:
        raise SystemExit("两份CSVviaprimary keymergeas空, Please 检查键列")
    if k2 in merged.columns and k2 != k1:
        merged.drop(columns=[k2], inplace=True)
    merged.rename(columns={k1: k_norm}, inplace=True)

    # Compute rule: use λem to infer λex
    ex_pred = pd.to_numeric(merged[ex_col], errors="coerce").astype(float).values
    em_pred = pd.to_numeric(merged[em_col], errors="coerce").astype(float).values
    A = float(args.A)
    B = float(args.B)
    
    # Inverse equation: λex_calc = (λem - B) / A
    ex_calc = (em_pred - B) / A

    ex_final = ex_pred.copy()
    mask_violate = em_pred < ex_pred  # physically unreasonable
    # Case 1: em < ex → use calculated ex from em
    ex_final[mask_violate] = ex_calc[mask_violate]
    # Case 2: em >= ex → average of calculated and predicted ex
    mask_ok = ~mask_violate & np.isfinite(ex_pred) & np.isfinite(ex_calc)
    ex_final[mask_ok] = 0.5 * (ex_calc[mask_ok] + ex_pred[mask_ok])

    # Output: keep both original and corrected values, plus the calculated intermediate
    merged["λex_pred_orig"] = ex_pred
    merged["λem_pred_orig"] = em_pred
    merged["λex_calc_from_em"] = ex_calc  # intermediate: calculated from em
    merged["λex_pred_shrink"] = ex_final  # final corrected ex
    merged["λem_pred_shrink"] = em_pred   # keep λem unchanged
    merged["n_fixed_flags"] = (merged.index * 0 + 1)

    # default output
    out = args.output
    if not out:
        p = Path(args.lem_csv)
        out = str(p.with_name(p.stem + "_constrained_affineYX_round1.csv"))

    merged.to_csv(out, index=False)
    n_fix = int(np.sum(mask_violate))
    print(f"[round1] Output: {out} | em<excorrected: {n_fix} | A={A}, B={B}")


if __name__ == "__main__":
    main()


