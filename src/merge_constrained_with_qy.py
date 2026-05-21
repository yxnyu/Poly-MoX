#!/usr/bin/env python3
import csv
import argparse
from typing import Dict, Any


def read_csv_to_dict(path: str, key_col: str = "Training data") -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if key_col not in reader.fieldnames:
            raise SystemExit(f"Key column '{key_col}' not found in {path}. headers: {reader.fieldnames}")
        for row in reader:
            k = row.get(key_col)
            if k is None or k == "":
                # skip malformed rows
                continue
            data[k] = row
    return data


def main():
    parser = argparse.ArgumentParser(description="Merge constrained λex/λem CSV with QY predictions by Training data key.")
    parser.add_argument("--constrained", required=True, help="path to constrained lex/lem CSV (with shrink columns)")
    parser.add_argument("--qy", required=True, help="path to QY prediction CSV")
    parser.add_argument("--output", required=True, help="path to write merged CSV")
    args = parser.parse_args()

    cons = read_csv_to_dict(args.constrained)
    qy = read_csv_to_dict(args.qy)

    # unified set of keys; preserve the order by sorted numeric if possible
    def _to_num(s: str):
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s

    keys = sorted(set(cons.keys()) | set(qy.keys()), key=_to_num)

    # prepare headers: union of columns, keep Training data first
    cons_headers = list(next(iter(cons.values())).keys()) if cons else []
    qy_headers = list(next(iter(qy.values())).keys()) if qy else []

    # Ensure Training data is the first column; then other columns from constrained, then those from qy not already present
    merged_headers = ["Training data"]
    for col in cons_headers:
        if col != "Training data" and col not in merged_headers:
            merged_headers.append(col)
    for col in qy_headers:
        if col != "Training data" and col not in merged_headers:
            merged_headers.append(col)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_headers)
        writer.writeheader()
        for k in keys:
            row = {h: "" for h in merged_headers}
            row["Training data"] = k
            if k in cons:
                for h, v in cons[k].items():
                    if h in merged_headers:
                        row[h] = v
            if k in qy:
                for h, v in qy[k].items():
                    if h in merged_headers:
                        row[h] = v
            writer.writerow(row)

    print(f"Merged {len(keys)} rows -> {args.output}")


if __name__ == "__main__":
    main()
