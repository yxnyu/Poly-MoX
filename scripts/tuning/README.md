# Optuna Hyperparameter Tuning

This folder contains a single generic launcher: **`tune_chain.sh`**.

It runs `src/moe_optuna.py` with:
- 5-fold CV objective (R²)
- SQLite-backed Optuna study (auto-resume across SLURM job kills)
- Automatic SLURM job chaining via `--dependency=afterany`

Each chain submission spawns N back-to-back 47-hour jobs; if any chain is
killed (wall-time, OOM, manual cancel), the next one resumes from the
SQLite study until `n_trials` is reached.

---

## Usage

```bash
./tune_chain.sh <tag> <radius> <n_bits> <use_init_pct> <csv> <n_trials> <study_name> [seed] [chain_count] [round_filter]
```

| Argument | Meaning | Example |
|----------|---------|---------|
| `tag` | Identifier; used for the SQLite file and the SLURM job name. Must contain `lex` / `lem` / `qy` so the target is auto-detected. | `95_lex` |
| `radius` | Morgan fingerprint radius | `2` |
| `n_bits` | Morgan fingerprint length | `2048` |
| `use_init_pct` | 0 or 1 — whether to add Initiator Percentage as numeric feature | `1` |
| `csv` | Training CSV path | `data/merged_final_classified.csv` |
| `n_trials` | Total Optuna trials | `<integer>` |
| `study_name` | Optuna study name in the SQLite DB | `95_lex` |
| `seed` *(default 42)* | KFold + Optuna seed | `42` |
| `chain_count` *(default 3)* | Number of chained 47h SLURM jobs | `3` |
| `round_filter` *(optional)* | Filter `Round` column to a single AL stage (e.g. `training_original` for the 95-sample subset) | `training_original` |

---

## Examples

### 1. 95-sample tuning (3 targets)

```bash
# Run from scripts/tuning/.  Filters Round=='training_original' from the 140-row file.
# Replace <N> with your trial budget.
./tune_chain.sh 95_lex 2 2048 1 data/merged_final_classified.csv <N> 95_lex 123 3 training_original
./tune_chain.sh 95_lem 2 2048 1 data/merged_final_classified.csv <N> 95_lem 42  3 training_original
./tune_chain.sh 95_qy  2 2048 1 data/merged_final_classified.csv <N> 95_qy  42  1 training_original
```

### 2. Use the 131-sample final-target training set

```bash
./tune_chain.sh target_lex 2 2048 1 data/target_training_131.csv <N> target_lex 42 3
./tune_chain.sh target_lem 2 2048 1 data/target_training_131.csv <N> target_lem 42 3
./tune_chain.sh target_qy  2 2048 1 data/target_training_131.csv <N> target_qy  42 3
```

---

## Outputs

- **SQLite study**: `optuna_studies/<tag>.db` — every trial is persisted, study can be resumed.
- **SLURM logs**: `logs/chain_<tag>/slurm-<jobid>.out`
- **Best checkpoint** (if `--save_best` is set inside): `runs/moe_optuna_<target>_<timestamp>/`

To extract the latest best_params from any in-progress / finished study:

```bash
python src/dump_study_best.py
# → writes best_params/optuna/<tag>_best.json
```

To verify reproducibility (re-run 5-fold OOF with the saved best_params):

```bash
sbatch ../reproduce/repro_tuned_params.sh
```
