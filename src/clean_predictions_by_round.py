#!/usr/bin/env python3
"""Filter repro prediction CSVs to keep only the IDs that match merged_final_classified.csv.

After this:
  - R1 predictions:  12 rows (Round == 'R1')
  - R2 predictions:  12 rows (Round == 'R2', drops ID 3560 which belongs to target)
  - R3 predictions:  12 rows (Round == 'R3', the 12 added to R4 training,
                              drops 6 batch3 samples that were never used)
  - R4 predictions:   9 rows (Round == 'target', drops ID 5239 which belongs to R3)

The original (legacy) prediction files are moved to results/predictions/legacy_full_eval_sets/
so the evaluation-set versions are still available.
"""
import csv
import shutil
from pathlib import Path

CV = Path(__file__).resolve().parent.parent
DATA = CV / "data" / "merged_final_classified.csv"
PRED_DIR = CV / "results" / "predictions"
LEGACY_DIR = PRED_DIR / "legacy_full_eval_sets"


def load_round_ids():
    """Load IDs per Round from merged_final_classified.csv."""
    rounds = {}
    with DATA.open() as f:
        for row in csv.DictReader(f):
            r = row.get("Round", "").strip()
            try:
                rid = int(row["Training Data"])
            except ValueError:
                continue
            rounds.setdefault(r, set()).add(rid)
    return rounds


def filter_csv(src: Path, keep_ids: set[int], dst: Path) -> int:
    """Filter rows in src to only those with Training Data in keep_ids; write to dst."""
    with src.open() as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(r["Training Data"]) in keep_ids]
    with dst.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main():
    rounds = load_round_ids()
    print(f"Round IDs loaded from {DATA.name}:")
    for r, ids in sorted(rounds.items()):
        print(f"  {r:<20} {len(ids):>4} IDs")

    LEGACY_DIR.mkdir(exist_ok=True)

    # Mapping: filename prefix → set of IDs to keep
    JOBS = [
        ("r1", "R1"),
        ("r2", "R2"),
        ("r3", "R3"),
        ("final_target", "final_target"),
    ]

    print("\nFiltering predictions...")
    for prefix, round_label in JOBS:
        keep_ids = rounds.get(round_label, set())
        for kind in ("lex", "lem", "qy"):
            patterns = [
                f"repro_{prefix}_pred_{kind}.csv",  # r1/r3 style
                f"repro_{prefix}_{kind}.csv",        # r2 style
            ]
            for pat in patterns:
                src = PRED_DIR / pat
                if not src.exists():
                    continue
                # Move original → legacy
                legacy_dst = LEGACY_DIR / src.name
                shutil.move(str(src), str(legacy_dst))
                # Write filtered new file
                new_name = src.name  # keep same filename
                kept = filter_csv(legacy_dst, keep_ids, PRED_DIR / new_name)
                print(f"  {src.name}: {kept} rows kept")


if __name__ == "__main__":
    main()
