# PolyMox

PolyMox is an active-learning framework (Poly-MoX) for data-efficient discovery
of clusteroluminescent polyesters that convert sunlight into plant-usable red
light for enhanced photosynthesis and sustainable agriculture.

Targets predicted: **λex** (excitation), **λem** (emission), **QY** (quantum yield).

Contents:

- MoE model: Morgan FP + ChemBERTa, each fed to XGBoost, fused by an ElasticNet gate
- 3 rounds of active learning (R1 → R3); training set grew 95 → 140 polymers
- Optuna hyperparameter tuning, SQLite-backed (auto-resumes across SLURM job kills)
- BO acquisition: Thompson Sampling, EI, UCB, Tanimoto diversity, phase-wise weighting

---

## Directory layout

```
clean_version/
├── README.md
│
├── src/                            # All Python source (16 modules)
│   ├── feature_utils.py            # Morgan fingerprints + ChemBERTa + feature alignment + cache
│   ├── regmix_core.py              # MoE model + Optuna search (SQLite-backed for resume)
│   ├── train_full_moe.py           # Train full MoE with fixed best_params
│   ├── predict_moe.py              # Load checkpoint and predict on a CSV
│   ├── moe_optuna.py               # 5-fold CV + Optuna driver (--storage flag)
│   ├── hyperparameter_search_moe.py# Fixed train/test split version of the driver
│   ├── eval_oof_95.py              # Reproduce 5-fold OOF metrics from a checkpoint
│   ├── extract_best_params.py      # Recover best_params from killed jobs' slurm-out
│   ├── dump_study_best.py          # Extract latest best_params from SQLite Optuna study
│   ├── classify_merged_final.py    # Tag the dataset with Round column
│   ├── clean_predictions_by_round.py # Filter predictions to align with merged_final
│   ├── bo_acquisition.py           # BO acquisition (TS/EI/UCB) + Tanimoto diversity
│   ├── apply_round1_affine_rule.py # Post-process: affine projection (λex ≤ a·λem + b)
│   ├── enforce_em_ge_ex.py         # Post-process: enforce λem ≥ λex
│   ├── merge_constrained_with_qy.py
│   └── make_final_from_merged.py
│
├── scripts/                        # SLURM submission scripts
│   ├── tuning/                     # Optuna hyperparameter tuning
│   │   ├── README.md
│   │   └── tune_chain.sh ⭐        # Generic parameterized chain launcher (auto-resume)
│   └── reproduce/
│       └── repro_tuned_params.sh   # Reproduce 5-fold CV metrics on the 95-sample dataset from saved best_params
│
├── data/  (2.2 MB)
│   ├── merged_final_classified.csv # All 140 labelled molecules with Round column
│   │                               #   (training_original=95, R1=12, R2=12, R3=12, final_target=9)
│   └── total_dataset.csv           # 11,531-molecule candidate pool for virtual screening
│
├── checkpoints/                    # Trained MoE checkpoints (unified r=2 b=2048)
│   ├── r1_{lex,lem,qy}             # R1 candidate scoring (n_samples=95, training_original only)
│   └── target_{lex,lem,qy}         # Final-target stage   (n_samples=131, RMSE-tuned)
│
├── best_params/                    # Optuna best hyperparameters
│   └── optuna/                     # 4 JSON files (3 targets × 95-sample + 95_lex seed=123)
│
└── results/
    └── predictions/                # Per-round MoE predictions, IDs aligned with merged_final.Round
        ├── repro_r1_pred_{lex,lem,qy}.csv             # 12 rows
        ├── repro_r2_{lex,lem,qy}.csv                  # 12 rows
        ├── repro_r3_pred_{lex,lem,qy}.csv             # 12 rows
        └── repro_final_target_pred_{lex,lem,qy}.csv   #  9 rows
```

The following directories are **gitignored** (regenerable / too large):
`logs/`, `optuna_studies/`, `runs/`, `.cache_features/`, `__pycache__/`, `slurm-*.out`.

---

## Environment

This repository is designed to run on an HPC cluster with SLURM + Singularity.

### Configuration via env vars

All SLURM scripts read these from the environment (no NYU/Greene-specific
paths are committed):

```bash
# Required — scripts abort with a clear error if any of these is unset:
export POLYMOX_SIMG=/path/to/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
export POLYMOX_OVERLAY=/path/to/overlay-15GB-500K.ext3:ro

# Optional:
export POLYMOX_ROOT=/path/to/your/repo/clone   # default: $PWD when sbatch is launched
export SBATCH_ACCOUNT=your_alloc               # SLURM reads this natively; or pass --account=foo to sbatch
```

The overlay ships Python 3.12 with the packages listed in `requirements.txt`
(numpy, pandas, scipy, scikit-learn, tqdm, rdkit, xgboost, torch+CUDA,
transformers, optuna). For `tune_chain.sh` specifically, `SLURM_ACCOUNT` is
also required (it is baked into the generated sbatch file).

### SLURM account

Configured in each SBATCH script as `#SBATCH --account=<your_account>`.

---

## Quick start

### 1. Run Optuna tuning with auto-resume

```bash
cd scripts/tuning
# Usage: ./tune_chain.sh <tag> <radius> <n_bits> <use_initiator_pct> <csv> <n_trials> <study_name> [seed] [chain_count] [round_filter]
./tune_chain.sh 95_lex 2 2048 1 data/merged_final_classified.csv <N_TRIALS> 95_lex 123 3 training_original
```

This submits 3 chained 47-hour SLURM jobs (`afterany` dependency). All trials are written
to `optuna_studies/<tag>.db` (SQLite); if any chain is killed, the next one resumes from
the database. The chain finishes when `n_trials` total are completed.

**Training-time physics constraint (default ON for λem).**
`tune_chain.sh` exports `ENFORCE_EM_GE_EX_TRAIN=1` with `EM_EX_CONSTRAINT_MODE=ratio`
and `RATIO_K=1.0`, so when the target is λem the training labels are clamped via
`y_em ← max(y_em, k·y_ex)` to satisfy the Stokes-shift inequality λem ≥ λex. Override
by exporting any of these env vars *before* `tune_chain.sh`:

```bash
export ENFORCE_EM_GE_EX_TRAIN=0          # disable
export EM_EX_CONSTRAINT_MODE=affine      # 'ratio' (default) or 'affine'
export RATIO_K=1.05                      # tighter Stokes shift margin
export AFFINE_A=0.844 AFFINE_B=12.0      # only used in 'affine' mode
```

See `scripts/tuning/README.md` for the full argument table and additional examples.

### 2. Extract current best params from a running / finished study

```bash
python src/dump_study_best.py
# → writes JSON files to best_params/optuna/
```

### 3. Reproduce the tuned metrics (5-fold OOF + strict nested CV)

```bash
sbatch scripts/reproduce/repro_tuned_params.sh
```

Prints two metrics per target:

- **Original-pipeline R² / PCC** — ElasticNet gate trained on the full OOF prediction matrix
  and evaluated on the same OOF (matches what `moe_optuna.py` optimised during search).
- **Strict nested-CV R² / PCC** — gate is also trained per-fold and predicts on the held-out
  fold only. Slightly more conservative but free of any leakage.

---

## Results

5-fold CV on 95 `training_original` polymers. Features: Morgan FP (r=2, 2048-bit)
+ ChemBERTa embedding + Initiator %. Optuna-tuned per target. Poly-MoX is the
MoE described above; baselines use the same feature configuration. Bold = best
column.

**Table 1 | λem**

| Method | Pearson r | R² | MAE (nm) | RMSE (nm) |
|---|---|---|---|---|
| **Poly-MoX** | **0.8950 ± 0.0437** | **0.8009 ± 0.0783** | **24.01 ± 2.46** | **33.61 ± 3.28** |
| Morgan + XGBoost | 0.8312 ± 0.0893 | 0.6908 ± 0.1484 | 31.85 ± 3.72 | 41.88 ± 5.16 |
| Morgan + RegMix | 0.8456 ± 0.0782 | 0.7150 ± 0.1323 | 29.42 ± 3.38 | 40.21 ± 4.63 |
| ChemBERTa + XGBoost | 0.6823 ± 0.1347 | 0.4655 ± 0.1838 | 42.17 ± 5.34 | 55.08 ± 7.21 |
| ChemBERTa + RegMix | 0.7012 ± 0.1263 | 0.4917 ± 0.1773 | 39.85 ± 4.87 | 53.71 ± 6.58 |
| Concat (ChemBERTa + Morgan) | 0.8102 ± 0.0951 | 0.6564 ± 0.1541 | 33.52 ± 4.13 | 44.16 ± 5.47 |

**Table 2 | λex**

| Method | Pearson r | R² | MAE (nm) | RMSE (nm) |
|---|---|---|---|---|
| **Poly-MoX** | **0.8211 ± 0.0614** | **0.6742 ± 0.1012** | **31.22 ± 2.83** | **41.04 ± 3.76** |
| Morgan + XGBoost | 0.7456 ± 0.1247 | 0.5559 ± 0.1863 | 38.67 ± 4.21 | 47.92 ± 5.89 |
| Morgan + RegMix | 0.7623 ± 0.1093 | 0.5811 ± 0.1667 | 36.54 ± 3.94 | 46.58 ± 5.32 |
| ChemBERTa + XGBoost | 0.6134 ± 0.1582 | 0.3762 ± 0.1941 | 48.23 ± 5.67 | 56.83 ± 7.43 |
| ChemBERTa + RegMix | 0.6387 ± 0.1438 | 0.4079 ± 0.1836 | 45.91 ± 5.13 | 55.39 ± 6.87 |
| Concat (ChemBERTa + Morgan) | 0.7289 ± 0.1176 | 0.5313 ± 0.1714 | 40.12 ± 4.58 | 49.28 ± 6.14 |

**Table 3 | QY**

| Method | Pearson r | R² | MAE | RMSE |
|---|---|---|---|---|
| **Poly-MoX** | **0.7745 ± 0.0893** | **0.5998 ± 0.1382** | **3.19 ± 0.31** | **4.64 ± 0.41** |
| Morgan + XGBoost | 0.6945 ± 0.1901 | 0.5023 ± 0.2641 | 3.48 ± 0.52 | 5.07 ± 0.73 |
| Morgan + RegMix | 0.7142 ± 0.1694 | 0.5100 ± 0.2418 | 3.41 ± 0.47 | 4.98 ± 0.65 |
| ChemBERTa + XGBoost | 0.4431 ± 0.1421 | 0.1963 ± 0.1532 | 5.03 ± 0.81 | 6.72 ± 1.12 |
| ChemBERTa + RegMix | 0.4602 ± 0.1356 | 0.1970 ± 0.1489 | 4.92 ± 0.74 | 6.57 ± 1.03 |
| Concat (ChemBERTa + Morgan) | 0.6952 ± 0.1753 | 0.4830 ± 0.2287 | 3.62 ± 0.55 | 5.21 ± 0.78 |

---

## Citation

If you use this pipeline, please cite the original paper (TBD).
