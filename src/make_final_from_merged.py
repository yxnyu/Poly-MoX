#!/usr/bin/env python3
import csv
import argparse
import math


def to_float(v):
    try:
        return float(v)
    except Exception:
        return math.nan


def main():
    ap = argparse.ArgumentParser(description="Make final predictions CSV: replace λex_pred/λem_pred with shrink values and drop helper columns.")
    ap.add_argument("--merged", required=True, help="path to merged CSV (with shrink/orig columns)")
    ap.add_argument("--output", required=True, help="path to write final CSV (without shrink/orig/n_fixed_flags)")
    args = ap.parse_args()

    with open(args.merged, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []

    # columns
    col_ex = "λex_pred" if "λex_pred" in headers else ("lex_pred" if "lex_pred" in headers else None)
    col_em = "λem_pred" if "λem_pred" in headers else ("lem_pred" if "lem_pred" in headers else None)
    col_ex_s = "λex_pred_shrink" if "λex_pred_shrink" in headers else ("lex_pred_shrink" if "lex_pred_shrink" in headers else None)
    col_em_s = "λem_pred_shrink" if "λem_pred_shrink" in headers else ("lem_pred_shrink" if "lem_pred_shrink" in headers else None)

    if not (col_ex and col_em and col_ex_s and col_em_s):
        raise SystemExit("required columns not found in merged CSV.")

    # override main preds with shrink values when available (prefer numeric)
    for r in rows:
        exs = to_float(r.get(col_ex_s, ""))
        ems = to_float(r.get(col_em_s, ""))
        if not math.isnan(exs):
            r[col_ex] = str(exs)
        if not math.isnan(ems):
            r[col_em] = str(ems)

    # build final headers: drop *_orig, *_shrink, n_fixed_flags
    drop_cols = set([
        "λex_pred_orig", "λem_pred_orig", "lex_pred_orig", "lem_pred_orig",
        "λex_pred_shrink", "λem_pred_shrink", "lex_pred_shrink", "lem_pred_shrink",
        "n_fixed_flags",
    ])
    final_headers = [h for h in headers if h not in drop_cols]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in final_headers})

    print(f"Wrote final CSV: {args.output} with {len(rows)} rows and {len(final_headers)} columns")


if __name__ == "__main__":
    main()
