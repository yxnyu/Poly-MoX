#!/usr/bin/env python3
"""from moe_optuna.py 的 slurm-out insalvagebest trial 的 params. """
import json, re, sys
from pathlib import Path

PARAMS_RE = re.compile(
    r"Trial (\d+) finished with value: ([0-9.eE+\-]+) and parameters: (\{[^}]+\})\."
)
BEST_RE = re.compile(r"Best is trial (\d+) with value: ([0-9.eE+\-]+)\.")


def extract(slurm_out: Path):
    best_r2 = -1e9
    best_trial = None
    best_params = None
    txt = slurm_out.read_text(errors="ignore")

    # go一遍all trials, 留belowmax的
    for m in PARAMS_RE.finditer(txt):
        trial_id = int(m.group(1))
        value = float(m.group(2))
        if value > best_r2:
            best_r2 = value
            best_trial = trial_id
            params_str = m.group(3)
            # Python dict syntax → JSON: 单引号 → 双引号
            try:
                best_params = json.loads(params_str.replace("'", '"'))
            except json.JSONDecodeError:
                best_params = {"raw": params_str}

    return best_trial, best_r2, best_params


def main():
    CV = Path(__file__).resolve().parent.parent
    salvage_dir = CV / "salvaged_best_params"
    salvage_dir.mkdir(exist_ok=True)

    JOBS = [
        ("8850138", "95_qy"),
        ("8861396", "95_lex"),
        ("8850177", "95_lem"),
    ]

    print(f"{'JobID':<10} {'tag':<25} {'BestTrial':<10} {'R²':<10} {'PCC≈'}")
    print("-" * 70)
    for jid, tag in JOBS:
        out = CV / f"slurm-{jid}.out"
        if not out.exists():
            print(f"{jid:<10} {tag:<25} (no slurm out)")
            continue
        trial, r2, params = extract(out)
        pcc = (r2 ** 0.5) if r2 > 0 else 0
        print(f"{jid:<10} {tag:<25} {trial:<10} {r2:<10.4f} {pcc:.4f}")
        if params:
            json_path = salvage_dir / f"{tag}_best.json"
            json_path.write_text(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
