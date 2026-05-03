#!/bin/bash
#SBATCH --job-name=massive-crossrep-ft
#SBATCH --account=PAS2836
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# =============================================================================
# MASSIVE Cross-Representation Finetuning
#
# Finetunes a text-pretrained model using romanized tokenizer and data.
# Tests whether phonetic representations transfer across representation types.
#
# Required environment variables:
#   SIZE          - Model size: "small" or "medium"
#   PRETRAINED    - Path to TEXT-pretrained checkpoint
#   TOKENIZER     - Path to ROMANIZED tokenizer JSON
#   OUT_ROOT      - Root output directory
#
# Optional:
#   LR            - Learning rate (default: 1e-4)
#   BATCH_SIZE    - Per-device batch size (default: 16)
#   GAS           - Gradient accumulation steps (default: 8)
#   NUM_EPOCHS    - Number of epochs (default: 3)
#   EVAL_INTERVAL - Steps between evaluations (default: 500)
#   ENV_PATH      - Path to conda environment
#   REPO_ROOT     - Path to this repository root
# =============================================================================

SIZE="${SIZE:?Must set SIZE (small or medium)}"
PRETRAINED="${PRETRAINED:?Must set PRETRAINED (text-pretrained) checkpoint path}"
TOKENIZER="${TOKENIZER:?Must set TOKENIZER (romanized tokenizer) path}"
OUT_ROOT="${OUT_ROOT:?Must set OUT_ROOT output directory}"

LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GAS="${GAS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
REPO_ROOT="${REPO_ROOT:-$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))}"
ENV_PATH="${ENV_PATH:-}"

echo "=========================================="
echo "MASSIVE CROSS-REP FT | ${SIZE^^} | text->romanized"
echo "BS=$((BATCH_SIZE * GAS)) | LR=${LR} | ${NUM_EPOCHS} epochs"
echo "Start: $(date)"
echo "Job ${SLURM_JOB_ID:-N/A} on ${SLURM_JOB_NODELIST:-local}"
echo "=========================================="

if [ -n "${ENV_PATH}" ]; then
    source "${ENV_PATH}/bin/activate"
else
    module load miniconda3/24.1.2-py310
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ipagpt_training
fi

for f in "${PRETRAINED}" "${TOKENIZER}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

RUN_ROOT="${OUT_ROOT}/cross_rep_seen/${SIZE}_text2rom"
DATA_DIR="${RUN_ROOT}/data_cache"
CKPT_DIR="${RUN_ROOT}/checkpoints"
HF_CACHE="${RUN_ROOT}/hf_cache"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${HF_CACHE}" "${LOG_DIR}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export XDG_CACHE_HOME="${HF_CACHE}"
export TORCH_HOME="${HF_CACHE}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

case "${SIZE}" in
    small)
        SCRIPT="${REPO_ROOT}/bundle/finetuning/massive/finetune_small.py"
        N_LAYER=12; N_HEAD=6; N_EMBD=768
        ;;
    medium)
        SCRIPT="${REPO_ROOT}/bundle/finetuning/massive/finetune_medium.py"
        N_LAYER=16; N_HEAD=8; N_EMBD=1024
        ;;
    *)
        echo "ERROR: Unknown SIZE=${SIZE}"; exit 1 ;;
esac

cd "${REPO_ROOT}"

python -u "${SCRIPT}" \
    --dataset "${SIZE}-text2rom-seen-all" \
    --pretrained_ckpt_path "${PRETRAINED}" \
    --tokenizer_name "${TOKENIZER}" \
    --hf_cache "${HF_CACHE}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${CKPT_DIR}" \
    --parent_dataset "mugezhang/massive_ipa_romanized" \
    --train_configs en-US es-ES hi-IN ur-PK ru-RU pl-PL ta-IN ml-IN \
    --eval_parent_langs en-US es-ES hi-IN ur-PK ru-RU pl-PL ta-IN ml-IN \
    --text_column romanized \
    --label_column intent \
    --num_classes 60 \
    --n_layer ${N_LAYER} --n_head ${N_HEAD} --n_embd ${N_EMBD} --block_size 1024 \
    --batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GAS} \
    --learning_rate ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --dtype bfloat16 \
    --eval_interval ${EVAL_INTERVAL} \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
