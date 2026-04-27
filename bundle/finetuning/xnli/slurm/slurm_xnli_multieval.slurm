#!/bin/bash
#SBATCH --job-name=xnli-multieval
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Required env vars:
#   SIZE        — small | medium | large_standard | xl
#   REP         — text | romanized | ipa_stripped
#   PRETRAINED  — absolute path to pretraining checkpoint .pt
#   TOKENIZER   — absolute path to tokenizer .json
#   TRAIN_BIN   — absolute path to multilingual train.bin
#   VAL_BIN_DIR — directory containing val_${lang}.bin per eval language
#   OUT_DIR     — output directory for FT checkpoints
#
# Optional (with sensible defaults):
#   EVAL_LANGUAGES (space-separated; default 8-lang set)
#   ENV_PATH       (conda env path; default shared env)
#   BATCH_SIZE GAS NUM_EPOCHS LR EVAL_INTERVAL EVAL_ITERS NUM_WORKERS
#   COSINE_LR      (1 to enable; default linear)
#   TORCH_COMPILE  (1 to enable; default off)
#   USE_CLASS_WEIGHTS (default true)
#   WANDB_LOG WANDB_PROJECT
#   DRY_RUN (1 to print and exit)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training}"
SIZE="${SIZE:?SIZE is required (small|medium|large_standard|xl)}"
REP="${REP:?REP is required (text|romanized|ipa_stripped)}"
PRETRAINED="${PRETRAINED:?PRETRAINED checkpoint path is required}"
TOKENIZER="${TOKENIZER:?TOKENIZER path is required}"
TRAIN_BIN="${TRAIN_BIN:?TRAIN_BIN path is required}"
VAL_BIN_DIR="${VAL_BIN_DIR:?VAL_BIN_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

EVAL_LANGUAGES="${EVAL_LANGUAGES:-english spanish russian polish tamil malayalam urdu hindi}"
ENV_PATH="${ENV_PATH:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/shared_envs/nanogpt_cu126}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GAS="${GAS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-1e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-}"  # empty → full val pass
NUM_WORKERS="${NUM_WORKERS:-4}"
COSINE_LR="${COSINE_LR:-0}"
TORCH_COMPILE="${TORCH_COMPILE:-0}"
USE_CLASS_WEIGHTS="${USE_CLASS_WEIGHTS:-true}"
WANDB_LOG="${WANDB_LOG:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ipa_xnli_multieval}"
DRY_RUN="${DRY_RUN:-0}"

case "${SIZE}" in
  small|medium|large_standard|xl) ;;
  *) echo "ERROR: bad SIZE=${SIZE}" >&2; exit 1 ;;
esac

case "${REP}" in
  text|romanized|ipa_stripped) ;;
  *) echo "ERROR: bad REP=${REP}" >&2; exit 1 ;;
esac

# Validate val bins exist for every requested eval language.
read -r -a EVAL_LANG_ARR <<< "${EVAL_LANGUAGES}"
VAL_BIN_ARGS=()
for lang in "${EVAL_LANG_ARR[@]}"; do
  bin="${VAL_BIN_DIR}/val_${lang}.bin"
  test -f "${bin}" || { echo "ERROR: missing val bin for ${lang}: ${bin}" >&2; exit 1; }
  VAL_BIN_ARGS+=(--val_bin "${lang}=${bin}")
done

test -f "${PRETRAINED}" || { echo "ERROR: PRETRAINED not found: ${PRETRAINED}" >&2; exit 1; }
test -f "${TOKENIZER}" || { echo "ERROR: TOKENIZER not found: ${TOKENIZER}" >&2; exit 1; }
test -f "${TRAIN_BIN}" || { echo "ERROR: TRAIN_BIN not found: ${TRAIN_BIN}" >&2; exit 1; }

DATASET_NAME="${SIZE}-${REP}-xnli"
JOB_OUT_DIR="${OUT_DIR}/${SIZE}/${REP}"
JOB_LOG_DIR="${JOB_OUT_DIR}/logs/${SLURM_JOB_ID:-local}"

mkdir -p "${JOB_OUT_DIR}" "${JOB_LOG_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/bundle/finetuning/xnli:${PYTHONPATH:-}"

module load miniconda3/24.1.2-py310 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -d "${ENV_PATH}" ]]; then
  conda activate "${ENV_PATH}"
else
  echo "ERROR: conda env path not found: ${ENV_PATH}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-local}"
if [[ "${WANDB_LOG}" != "1" ]]; then
  export WANDB_MODE=disabled
fi

FINETUNE_ARGS=(
  --size "${SIZE}"
  --dataset "${DATASET_NAME}"
  --pretrained_ckpt_path "${PRETRAINED}"
  --tokenizer_name "${TOKENIZER}"
  --train_bin "${TRAIN_BIN}"
  "${VAL_BIN_ARGS[@]}"
  --eval_languages "${EVAL_LANG_ARR[@]}"
  --out_dir "${JOB_OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GAS}"
  --num_epochs "${NUM_EPOCHS}"
  --learning_rate "${LR}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_workers "${NUM_WORKERS}"
  --use_class_weights "${USE_CLASS_WEIGHTS}"
  --wandb_project "${WANDB_PROJECT}"
)

if [[ -n "${EVAL_ITERS}" ]]; then
  FINETUNE_ARGS+=(--eval_iters "${EVAL_ITERS}")
fi
if [[ "${COSINE_LR}" == "1" ]]; then
  FINETUNE_ARGS+=(--cosine_lr)
fi
if [[ "${TORCH_COMPILE}" == "1" ]]; then
  FINETUNE_ARGS+=(--torch_compile)
fi
if [[ "${WANDB_LOG}" == "1" ]]; then
  FINETUNE_ARGS+=(--wandb_log)
fi

echo "=========================================="
echo "XNLI multieval | ${SIZE} | ${REP} | ${EVAL_LANGUAGES}"
echo "Pretrained: ${PRETRAINED}"
echo "Tokenizer:  ${TOKENIZER}"
echo "Train bin:  ${TRAIN_BIN}"
echo "Val dir:    ${VAL_BIN_DIR}"
echo "Out dir:    ${JOB_OUT_DIR}"
echo "Effective batch size: $((BATCH_SIZE * GAS))"
echo "LR ${LR} | LR_schedule=$( [[ ${COSINE_LR} == 1 ]] && echo cosine || echo linear )"
echo "torch.compile: ${TORCH_COMPILE} | use_class_weights: ${USE_CLASS_WEIGHTS}"
echo "Start: $(date)"
echo "=========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${FINETUNE_ARGS[@]}"
  exit 0
fi

# Forward stop signals from slurm to the python process.
TRAIN_PID=""
forward_signal() {
  local sig="$1"
  if [[ -n "${TRAIN_PID}" ]]; then
    echo "Forwarding ${sig} to training process ${TRAIN_PID}" >&2
    kill "-${sig}" "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

python -u "${REPO_ROOT}/bundle/finetuning/xnli/finetune_multieval.py" "${FINETUNE_ARGS[@]}" \
  > >(tee "${JOB_LOG_DIR}/train.log") 2>&1 &
TRAIN_PID=$!

set +e
wait "${TRAIN_PID}"
train_status=$?
set -e

trap - USR1 TERM

echo "=========================================="
echo "Done: $(date) | exit=${train_status}"
echo "=========================================="
exit "${train_status}"
