#!/usr/bin/env python3
"""
Classify merged_final.csv by active-learning round and add Round column. 
classificationrule: 
  - training_original (ID 1~95)       : 95 个original training data
  - R1 (training-set augmentation)                   : R1 选定的 12 个 training_add
  - R2_test (Batch2 test set)           : R2 stage12 test samples picked from the candidate pool
  - R3_test (Batch3 test set)           : R3 stage18 test samples picked from the candidate pool
  - R3_best12 (best-12 subset of Batch3) : 12 samples with highest PCC picked from R3 test set
  - R4_extra (newly added in the final stage)         : merged_final 9 samples without a Type label
"""
import argparse
import csv
from pathlib import Path

# IDs added to the training set in each round (derived from training_r2/r3/final.csv set diff)
# R1: training_r2 - training_original  (95→107, new增 12)
R1 = {288, 3569, 3606, 3612, 4747, 5112,
      5140, 5184, 8329, 8471, 10860, 11638}

# R2: training_r3 - training_r2  (107→119, new增 12, 但 ID=1 已at original in)
R2 = {1, 2981, 3160, 3493, 4083, 5772,
      6498, 6702, 8194, 8435, 8459, 9962}

# R3: training_final - training_r3  (119→131, new增 12)
R3 = {1903, 3521, 3527, 3624, 4819, 5208,
      5952, 6696, 7915, 8350, 8448, 11688}


def classify(id_val: int, type_str: str) -> str:
    """根据 ID andoriginal Type decide Round 标record"""
    if type_str == "training_original":
        return "training_original"
    if id_val in R1:
        return "R1"
    if id_val in R2:
        return "R2"
    if id_val in R3:
        return "R3"
    return "final_target"  # newly added in the final stage的experimentdata (最终PredictTarget)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Tag merged_final.csv rows with their AL Round")
    ap.add_argument("--input", type=Path, required=True, help="Input merged_final.csv (untagged source)")
    ap.add_argument("--output", type=Path, default=repo_root / "data" / "merged_final_classified.csv",
                    help="Output CSV with Round column (default: data/merged_final_classified.csv)")
    args = ap.parse_args()
    src, dst = args.input, args.output

    with src.open() as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)

    new_cols = ["Round"] + cols
    counter = {}
    for r in rows:
        try:
            id_val = int(r["Training data"])
        except (ValueError, KeyError):
            id_val = -1
        r["Round"] = classify(id_val, r.get("Type", "").strip())
        counter[r["Round"]] = counter.get(r["Round"], 0) + 1

    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to {dst}")
    print(f"total rows: {len(rows)}\n")

    print(f"{'Round':<20}{'count':<6}")
    print("-" * 30)
    order = ["training_original", "R1", "R2", "R3", "final_target"]
    for k in order:
        if k in counter:
            print(f"{k:<20}{counter[k]:<6}")


if __name__ == "__main__":
    main()
