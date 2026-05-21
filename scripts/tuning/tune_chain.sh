#!/bin/bash -e
# tune_chain.sh — Optuna hyperparameter tuning on SLURM with auto-resume.
#
# Uses SQLite storage so an Optuna study can resume across SLURM jobs. When a
# job hits its wall-time and is killed, the next chained job (--dependency=afterany)
# loads the same study DB and continues until n_trials is reached.
#
# Usage:
#   ./tune_chain.sh <tag> <radius> <n_bits> <use_init_pct 0|1> <csv> <n_trials> <study_name> [seed] [chain_count] [round_filter]
#
# Required env vars (this script will refuse to submit without them):
#   POLYMOX_SIMG    — absolute path to your Singularity .sif image
#   POLYMOX_OVERLAY — absolute path to your overlay file (e.g. .ext3:ro)
#   SLURM_ACCOUNT   — your cluster allocation name
# Optional:
#   POLYMOX_ROOT    — repo root on the cluster (default: $PWD)
#
# Example:
#   export POLYMOX_SIMG=/path/to/cuda12.3.2.sif
#   export POLYMOX_OVERLAY=/path/to/overlay-15GB-500K.ext3:ro
#   export SLURM_ACCOUNT=your_alloc
#   ./tune_chain.sh 95_lex 2 2048 1 data/merged_final_classified.csv 10000 95_lex 42 3 training_original

: "${POLYMOX_SIMG:?Set POLYMOX_SIMG to your Singularity image path}"
: "${POLYMOX_OVERLAY:?Set POLYMOX_OVERLAY to your overlay file path}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your cluster allocation}"

TGT_TAG=$1        # e.g. 95_lex → used for study filename and SLURM job name
RADIUS=${2:-2}
NBITS=${3:-2048}
USE_PCT=${4:-1}
CSV=${5:-data/merged_final_classified.csv}
NTRIALS=${6:-10000}
STUDY_NAME=${7:-$TGT_TAG}
SEED=${8:-42}
CHAIN=${9:-3}                # chain length (each ≤47h; 3 chains = up to 141h wall-time)
ROUND_FILTER=${10:-}         # Optional: filter Round column (e.g. 'training_original' for the 95-sample subset)

# Auto-detect target column from tag substring
if   echo "$TGT_TAG" | grep -qi "lex"; then TARGET="λex"
elif echo "$TGT_TAG" | grep -qi "lem"; then TARGET="λem"
elif echo "$TGT_TAG" | grep -qi "qy";  then TARGET="QY"
else
    echo "ERROR: tag '$TGT_TAG' must contain 'lex', 'lem', or 'qy' so the target can be inferred." >&2
    exit 1
fi

WORKDIR="${POLYMOX_ROOT:-$PWD}"

# Training-time physics constraint for the λem target.
# Default: ratio mode with k=1.0 → clamp y_em ≥ y_ex (Stokes shift ≥ 0).
# Override any of these BEFORE running tune_chain.sh to disable or change mode.
TRAIN_ENFORCE="${ENFORCE_EM_GE_EX_TRAIN:-1}"
TRAIN_MODE="${EM_EX_CONSTRAINT_MODE:-ratio}"   # 'ratio' or 'affine'
TRAIN_RATIO_K="${RATIO_K:-1.0}"
TRAIN_AFFINE_A="${AFFINE_A:-0.844}"
TRAIN_AFFINE_B="${AFFINE_B:-12.0}"
STUDY_DB="${WORKDIR}/optuna_studies/${TGT_TAG}.db"
LOG_DIR="${WORKDIR}/logs/chain_${TGT_TAG}"
mkdir -p "${WORKDIR}/optuna_studies" "${LOG_DIR}" "${WORKDIR}/runs"

cat > /tmp/tune_${TGT_TAG}.sbatch <<EOSH
#!/bin/bash -e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=t_${TGT_TAG}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --output=${LOG_DIR}/slurm-%j.out

singularity exec --nv --overlay "${POLYMOX_OVERLAY}" "${POLYMOX_SIMG}" /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export HF_HOME=/ext3/hf_cache
    export PYTHONUSERBASE=/ext3/python_packages
    export PATH=\\\$PYTHONUSERBASE/bin:\\\$PATH
    export PYTHONPATH=\\\$PYTHONUSERBASE/lib/python3.12/site-packages:\\\$PYTHONPATH:${WORKDIR}/src
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export WITH_INITIATOR_PCT=${USE_PCT}
    export ENFORCE_EM_GE_EX_TRAIN=${TRAIN_ENFORCE}
    export EM_EX_CONSTRAINT_MODE=${TRAIN_MODE}
    export RATIO_K=${TRAIN_RATIO_K}
    export AFFINE_A=${TRAIN_AFFINE_A}
    export AFFINE_B=${TRAIN_AFFINE_B}
    cd ${WORKDIR}

    echo '═══ ${TGT_TAG} resume job (study=${STUDY_NAME})═══'
    python src/moe_optuna.py \\
        --csv ${CSV} \\
        --target '${TARGET}' \\
        --hf_model DeepChem/ChemBERTa-77M-MLM \\
        --device cuda \\
        --radius ${RADIUS} \\
        --n_bits ${NBITS} \\
        --optuna_trials ${NTRIALS} \\
        --cache_dir .cache_features \\
        --save_best \\
        --save_dir runs \\
        --seed ${SEED} \\
        --storage 'sqlite:///${STUDY_DB}' \\
        --study_name '${STUDY_NAME}'${ROUND_FILTER:+ \\
        --round_filter '${ROUND_FILTER}'}
"
EOSH

# Chain submit: job N depends on N-1 (resumes whether N-1 succeeded or failed)
PREV=""
JOBS=()
for i in $(seq 1 $CHAIN); do
    if [ -z "$PREV" ]; then
        J=$(sbatch --parsable /tmp/tune_${TGT_TAG}.sbatch)
    else
        J=$(sbatch --parsable --dependency=afterany:${PREV} /tmp/tune_${TGT_TAG}.sbatch)
    fi
    JOBS+=($J)
    PREV=$J
    echo "  chain[$i]: $J"
done

echo ""
echo "✓ $TGT_TAG: Submitted $CHAIN chained jobs, study DB=${STUDY_DB}"
echo "  job chain: ${JOBS[@]}"
