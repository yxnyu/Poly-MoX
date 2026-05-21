#!/usr/bin/env python3
"""from SQLite Optuna study extract latest best_params and best_value, Saveas JSON. 

Usage:
  python dump_study_best.py
"""
import json
import optuna
from pathlib import Path

CV = Path(__file__).resolve().parent.parent
DB_DIR = CV / "optuna_studies"
OUT_DIR = CV / "best_params_from_study"
OUT_DIR.mkdir(exist_ok=True)

STUDIES = ["95_lex", "95_lem", "95_qy"]  # convention: study_name == tag == DB basename

print(f"{'tag':<25}{'n_trials':<10}{'best_r2':<10}{'pcc≈':<10}")
print("-" * 60)

for tag in STUDIES:
    study_name = tag
    db = DB_DIR / f"{tag}.db"
    if not db.exists():
        print(f"{tag:<25} (no db)")
        continue
    try:
        study = optuna.load_study(
            storage=f"sqlite:///{db}",
            study_name=study_name,
        )
        n = len(study.trials)
        if n == 0:
            print(f"{tag:<25} {n:<10} (空 study)")
            continue
        r2 = study.best_value
        pcc = r2 ** 0.5 if r2 > 0 else 0
        print(f"{tag:<25} {n:<10} {r2:<10.4f} {pcc:<10.4f}")

        # Save best params with reproducibility metadata at the top
        meta = {k: study.user_attrs[k] for k in ("seed", "target", "n_samples") if k in study.user_attrs}
        annotated = {"_meta": meta, **study.best_params} if meta else dict(study.best_params)
        (OUT_DIR / f"{tag}_best.json").write_text(
            json.dumps(annotated, indent=2, ensure_ascii=False) + "\n"
        )
    except Exception as e:
        print(f"{tag:<25} Readfailed: {e}")

print()
print(f"reproduction: {OUT_DIR}")
