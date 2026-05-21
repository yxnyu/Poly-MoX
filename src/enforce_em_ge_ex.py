#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from feature_utils import normalize_column_name, find_smiles_column


def pick_join_key(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    # Prefer Training data
    cols1 = {normalize_column_name(c): c for c in df1.columns}
    cols2 = {normalize_column_name(c): c for c in df2.columns}
    if "Training data" in cols1 and "Training data" in cols2:
        return cols1["Training data"], cols2["Training data"], "Training data"
    # Fallback to SMILES-like column
    key1 = find_smiles_column(list(df1.columns))
    key2 = find_smiles_column(list(df2.columns))
    return key1, key2, normalize_column_name(key1)


def project_to_ratio_line(x0: np.ndarray, y0: np.ndarray, k: float, w_ex: float = 1.0, w_em: float = 1.0):
    """Project (x0,y0) to the line y = k*x (weighted least squares). Returns (x*, y*)."""
    # Minimize w_ex*(x-x0)^2 + w_em*(y-y0)^2 s.t. y = k x
    # Substitute y: f(x) = w_ex*(x-x0)^2 + w_em*(k x - y0)^2
    # df/dx = 2 w_ex (x-x0) + 2 w_em k (k x - y0) = 0
    # x* = (w_ex x0 + w_em k y0) / (w_ex + w_em k^2)
    denom = (w_ex + w_em * (k ** 2))
    x_star = (w_ex * x0 + w_em * k * y0) / denom
    y_star = k * x_star
    return x_star, y_star


def project_to_affine_line(x0: np.ndarray, y0: np.ndarray, a: float, b: float, w_ex: float = 1.0, w_em: float = 1.0):
    """Project (x0,y0) to the line x = a*y + b under weighted L2 metric.
    Parameterize with t=y: minimize w_ex*(a t + b - x0)^2 + w_em*(t - y0)^2.
    t* = (w_ex*a*(x0 - b) + w_em*y0) / (w_ex*a^2 + w_em)
    """
    denom = (w_ex * (a ** 2) + w_em)
    t_star = (w_ex * a * (x0 - b) + w_em * y0) / denom
    y_star = t_star
    x_star = a * t_star + b
    return x_star, y_star


def project_to_affine_line_yx(x0: np.ndarray, y0: np.ndarray, A: float, B: float, w_ex: float = 1.0, w_em: float = 1.0):
    """Project (x0,y0) to the line y = A*x + B under weighted L2 metric.
    Minimize w_ex*(x-x0)^2 + w_em*(A x + B - y0)^2.
    x* = (w_ex*x0 + w_em*A*(y0 - B)) / (w_ex + w_em*A^2)
    y* = A*x* + B
    """
    denom = (w_ex + w_em * (A ** 2))
    x_star = (w_ex * x0 + w_em * A * (y0 - B)) / denom
    y_star = A * x_star + B
    return x_star, y_star


def main():
    ap = argparse.ArgumentParser(description="Merge two predictions and enforce constraints with optional linear shrink")
    ap.add_argument("--lex_csv", required=True, help="path to jianyu_prediction_pred_lex.csv")
    ap.add_argument("--lem_csv", required=True, help="path to jianyu_prediction_pred_lem.csv")
    # Ratio constraint options
    ap.add_argument("--ratio", type=float, default=None, help="k in λem ≥ k-λex; if set with --use_ratio, enforce boundary; if also --shrink_both, project to y=kx")
    ap.add_argument("--use_ratio", action="store_true", help="Enable ratio constraint handling")
    ap.add_argument("--shrink_both", action="store_true", help="When using ratio, project both (λex,λem) to the line y=kx instead of only lifting λem")
    # Affine constraint options: ex = a*em + b
    ap.add_argument("--affine_a", type=float, default=None, help="Affine a in ex = a*em + b (e.g., 0.844)")
    ap.add_argument("--affine_b", type=float, default=None, help="Affine b in ex = a*em + b (e.g., -12)")
    ap.add_argument("--affine_enforce", choices=["eq", "le", "ge"], default="eq", help="eq: always project to the line; le: only if ex > a*em + b; ge: only if ex < a*em + b")
    # Affine constraint options: y = A*x + B
    ap.add_argument("--affine_yx_a", type=float, default=None, help="Affine A in λem = A*λex + B")
    ap.add_argument("--affine_yx_b", type=float, default=None, help="Affine B in λem = A*λex + B")
    ap.add_argument("--affine_yx_enforce", choices=["eq", "le", "ge"], default="eq", help="For y = A*x + B: eq=always, le: only if y > A*x + B, ge: only if y < A*x + B")
    ap.add_argument("--apply_when_ex_gt_em", action="store_true", help="When using affine y=A*x+B with eq mode, only apply on rows where λex > λem")
    # weights for shrink (trust one axis more)
    ap.add_argument("--w_ex", type=float, default=1.0, help="weight for λex deviation in projection")
    ap.add_argument("--w_em", type=float, default=1.0, help="weight for λem deviation in projection")
    # IO
    ap.add_argument("--output", required=False, help="Output merged CSV path")
    args = ap.parse_args()

    df_ex = pd.read_csv(args.lex_csv)
    df_em = pd.read_csv(args.lem_csv)

    # Normalize names and locate columns
    map_ex = {c: normalize_column_name(c) for c in df_ex.columns}
    map_em = {c: normalize_column_name(c) for c in df_em.columns}
    df_ex.rename(columns=map_ex, inplace=True)
    df_em.rename(columns=map_em, inplace=True)

    ex_col = None
    em_col = None
    for c in df_ex.columns:
        if normalize_column_name(c) == "λex_pred":
            ex_col = c
            break
    for c in df_em.columns:
        if normalize_column_name(c) == "λem_pred":
            em_col = c
            break
    if ex_col is None or em_col is None:
        raise ValueError("not yetcanatInputCSVinfound λex_pred / λem_pred 列")

    key1, key2, key_norm = pick_join_key(df_ex, df_em)
    merged = df_ex[[key1, ex_col]].merge(
        df_em[[key2, em_col]], left_on=key1, right_on=key2, how="inner",
        suffixes=("_ex", "_em")
    )
    if merged.empty:
        raise ValueError("两份CSVno法viaprimary keymerge, Please confirm二者来源consistent")
    if key2 in merged.columns and key2 != key1:
        merged.drop(columns=[key2], inplace=True)
    merged.rename(columns={key1: key_norm}, inplace=True)

    # Parse arrays
    x = pd.to_numeric(merged[ex_col], errors="coerce").astype(float).values  # λex
    y = pd.to_numeric(merged[em_col], errors="coerce").astype(float).values  # λem
    x_shrink = x.copy()
    y_shrink = y.copy()

    n_fix = 0
    # 1) Ratio constraint handling
    if args.use_ratio and args.ratio is not None:
        k = float(args.ratio)
        if args.shrink_both:
            # Project to y = k x only when violated (y < kx)
            violated = (y < k * x)
            if np.any(violated):
                xs, ys = project_to_ratio_line(x[violated], y[violated], k, args.w_ex, args.w_em)
                x_shrink[violated] = xs
                y_shrink[violated] = ys
                n_fix += int(np.sum(violated))
        else:
            # Only lift λem up to boundary, keep λex
            violated = (y < k * x)
            y_shrink[violated] = k * x[violated]
            n_fix += int(np.sum(violated))

    # 2) Affine constraint handling: ex = a*em + b
    if args.affine_a is not None and args.affine_b is not None:
        a = float(args.affine_a)
        b = float(args.affine_b)
        # Decide which rows need projection
        if args.affine_enforce == "eq":
            need = np.isfinite(x_shrink) & np.isfinite(y_shrink)
        elif args.affine_enforce == "le":
            # enforce x <= a*y + b
            need = x_shrink > (a * y_shrink + b)
        else:  # ge
            # enforce x >= a*y + b
            need = x_shrink < (a * y_shrink + b)
        if np.any(need):
            xs, ys = project_to_affine_line(x_shrink[need], y_shrink[need], a, b, args.w_ex, args.w_em)
            x_shrink[need] = xs
            y_shrink[need] = ys
            n_fix += int(np.sum(need))

    # 3) Affine constraint handling: y = A*x + B (e.g., λem = A*λex + B)
    if args.affine_yx_a is not None and args.affine_yx_b is not None:
        A = float(args.affine_yx_a)
        B = float(args.affine_yx_b)
        # Decide which rows need projection
        if args.affine_yx_enforce == "eq":
            need = np.isfinite(x_shrink) & np.isfinite(y_shrink)
            if args.apply_when_ex_gt_em:
                need = need & (x_shrink > y_shrink)
        elif args.affine_yx_enforce == "le":
            # enforce y <= A*x + B
            need = y_shrink > (A * x_shrink + B)
        else:  # ge
            # enforce y >= A*x + B
            need = y_shrink < (A * x_shrink + B)
        if np.any(need):
            xs, ys = project_to_affine_line_yx(x_shrink[need], y_shrink[need], A, B, args.w_ex, args.w_em)
            x_shrink[need] = xs
            y_shrink[need] = ys
            n_fix += int(np.sum(need))

    # Write outputs with both original and shrinked values
    merged["λex_pred_orig"] = x
    merged["λem_pred_orig"] = y
    merged["λex_pred_shrink"] = x_shrink
    merged["λem_pred_shrink"] = y_shrink
    merged["n_fixed_flags"] = (merged.index * 0 + 1)  # placeholder to keep consistent schema

    out_path = args.output
    if not out_path:
        # default file name based on selected options
        p = Path(args.lem_csv)
        tag = []
        if args.use_ratio and args.ratio is not None:
            tag.append(f"ratio{k}")
            tag.append("both" if args.shrink_both else "lift_em")
        if args.affine_a is not None and args.affine_b is not None:
            tag.append(f"affine_{args.affine_a}_{args.affine_b}_{args.affine_enforce}")
        if args.affine_yx_a is not None and args.affine_yx_b is not None:
            yx_tag = f"affineYX_{args.affine_yx_a}_{args.affine_yx_b}_{args.affine_yx_enforce}"
            if args.apply_when_ex_gt_em:
                yx_tag += "_exgt"
            tag.append(yx_tag)
        suffix = "_" + "_".join(tag) if tag else ""
        out_path = str(p.with_name(p.stem + f"_constrained{suffix}.csv"))
    merged.to_csv(out_path, index=False)
    print(f"✅ Outputsaved: {out_path} | constraint/收缩processitemsitems (possibly重叠计数): {n_fix}")


if __name__ == "__main__":
    main()
