#!/bin/bash
# Submit XNLI multieval fine-tuning jobs across (size, rep) combinations.
#
#     bash submit_xnli_multieval.sh
#     SIZES="small medium" bash submit_xnli_multieval.sh
#     REPS="text" bash submit_xnli_multieval.sh
#     DRY_RUN=1 bash submit_xnli_multieval.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training}"
SLURM_SCRIPT="${REPO_ROOT}/bundle/finetuning/xnli/slurm/slurm_xnli_multieval.slurm"

# Path templates: __SIZE__ / __REP__ / __REP_ALIAS__ are substituted at submit time.
DEFAULT_CKPT_DIR_TEMPLATE="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/__REP__/__SIZE__"
DEFAULT_TOKENIZER_TEMPLATE="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang___REP__/bpe-8lang-__REP_ALIAS__-100k-tokenizer.json"
DEFAULT_TRAIN_BIN_TEMPLATE="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/xnli_bins/__REP__/train.bin"
DEFAULT_VAL_BIN_DIR_TEMPLATE="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/xnli_bins/__REP__"
DEFAULT_OUT_BASE="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/xnli_multieval_runs"

CKPT_DIR_TEMPLATE="${CKPT_DIR_TEMPLATE:-$DEFAULT_CKPT_DIR_TEMPLATE}"
TOKENIZER_TEMPLATE="${TOKENIZER_TEMPLATE:-$DEFAULT_TOKENIZER_TEMPLATE}"
TRAIN_BIN_TEMPLATE="${TRAIN_BIN_TEMPLATE:-$DEFAULT_TRAIN_BIN_TEMPLATE}"
VAL_BIN_DIR_TEMPLATE="${VAL_BIN_DIR_TEMPLATE:-$DEFAULT_VAL_BIN_DIR_TEMPLATE}"
OUT_BASE="${OUT_BASE:-$DEFAULT_OUT_BASE}"

# Slurm / cluster
ACCOUNT="${ACCOUNT:-PAS2836}"
PARTITION="${PARTITION:-gpu}"
ENV_PATH="${ENV_PATH:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/shared_envs/nanogpt_cu126}"
WALLTIME="${WALLTIME:-24:00:00}"

# Training defaults (per size; override via env to customize)
BATCH_SIZE="${BATCH_SIZE:-16}"
GAS="${GAS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-1e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-}"
COSINE_LR="${COSINE_LR:-0}"
TORCH_COMPILE="${TORCH_COMPILE:-0}"
USE_CLASS_WEIGHTS="${USE_CLASS_WEIGHTS:-true}"
WANDB_LOG="${WANDB_LOG:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ipa_xnli_multieval}"
EVAL_LANGUAGES="${EVAL_LANGUAGES:-english spanish russian polish tamil malayalam urdu hindi}"

# Job matrix
SIZES="${SIZES:-small medium large_standard xl}"
REPS="${REPS:-text romanized ipa_stripped}"
DRY_RUN="${DRY_RUN:-0}"

# ----------------- end PATH CONFIG -----------------

declare -A REP_ALIAS=(
  [text]=text
  [romanized]=romanized
  [ipa_stripped]=ipa-stripped
)

# Per-size GPU count (override via env if needed)
declare -A SIZE_GPUS=(
  [small]=1
  [medium]=1
  [large_standard]=1
  [xl]=1
)

resolve_template() {
  local tmpl="$1" size="$2" rep="$3"
  local rep_alias="${REP_ALIAS[$rep]:-$rep}"
  echo "$tmpl" | sed -e "s|__SIZE__|${size}|g" -e "s|__REP__|${rep}|g" -e "s|__REP_ALIAS__|${rep_alias}|g"
}

resolve_ckpt() {
  local size="$1" rep="$2"
  local ckpt_dir
  ckpt_dir="$(resolve_template "${CKPT_DIR_TEMPLATE}" "${size}" "${rep}")"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "ERROR: checkpoint dir not found: ${ckpt_dir}" >&2
    return 1
  fi
  local ckpt
  ckpt="$(ls -1 "${ckpt_dir}"/000_*/best_state_*.pt 2>/dev/null | tail -n1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: no best_state_*.pt under ${ckpt_dir}/000_*/" >&2
    return 1
  fi
  echo "${ckpt}"
}

mkdir -p "${OUT_BASE}"

submitted=()
errors=()

for size in ${SIZES}; do
  for rep in ${REPS}; do
    pretrained="$(resolve_ckpt "${size}" "${rep}")" || { errors+=("${size}/${rep}: ckpt"); continue; }
    tokenizer="$(resolve_template "${TOKENIZER_TEMPLATE}" "${size}" "${rep}")"
    train_bin="$(resolve_template "${TRAIN_BIN_TEMPLATE}" "${size}" "${rep}")"
    val_bin_dir="$(resolve_template "${VAL_BIN_DIR_TEMPLATE}" "${size}" "${rep}")"

    for path in "${tokenizer}" "${train_bin}"; do
      if [[ ! -f "${path}" ]]; then
        errors+=("${size}/${rep}: missing ${path}")
        continue 2
      fi
    done

    if [[ ! -d "${val_bin_dir}" ]]; then
      errors+=("${size}/${rep}: missing val dir ${val_bin_dir}")
      continue
    fi

    job_name="xnli-${size}-${rep}-multieval"
    gpus="${SIZE_GPUS[$size]:-1}"

    sbatch_args=(
      --account="${ACCOUNT}"
      --partition="${PARTITION}"
      --gpus-per-node="${gpus}"
      --time="${WALLTIME}"
      --job-name="${job_name}"
      --output="${OUT_BASE}/${size}/${rep}/logs/%x-%j.out"
      --error="${OUT_BASE}/${size}/${rep}/logs/%x-%j.err"
    )

    mkdir -p "${OUT_BASE}/${size}/${rep}/logs"

    env_export="ALL"
    env_export+=",REPO_ROOT=${REPO_ROOT}"
    env_export+=",SIZE=${size}"
    env_export+=",REP=${rep}"
    env_export+=",PRETRAINED=${pretrained}"
    env_export+=",TOKENIZER=${tokenizer}"
    env_export+=",TRAIN_BIN=${train_bin}"
    env_export+=",VAL_BIN_DIR=${val_bin_dir}"
    env_export+=",OUT_DIR=${OUT_BASE}"
    env_export+=",ENV_PATH=${ENV_PATH}"
    env_export+=",EVAL_LANGUAGES=${EVAL_LANGUAGES}"
    env_export+=",BATCH_SIZE=${BATCH_SIZE}"
    env_export+=",GAS=${GAS}"
    env_export+=",NUM_EPOCHS=${NUM_EPOCHS}"
    env_export+=",LR=${LR}"
    env_export+=",EVAL_INTERVAL=${EVAL_INTERVAL}"
    env_export+=",EVAL_ITERS=${EVAL_ITERS}"
    env_export+=",COSINE_LR=${COSINE_LR}"
    env_export+=",TORCH_COMPILE=${TORCH_COMPILE}"
    env_export+=",USE_CLASS_WEIGHTS=${USE_CLASS_WEIGHTS}"
    env_export+=",WANDB_LOG=${WANDB_LOG}"
    env_export+=",WANDB_PROJECT=${WANDB_PROJECT}"
    sbatch_args+=(--export="${env_export}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY] sbatch ${sbatch_args[*]} ${SLURM_SCRIPT}"
      echo "      pretrained=${pretrained}"
      continue
    fi

    jid=$(sbatch --parsable "${sbatch_args[@]}" "${SLURM_SCRIPT}")
    submitted+=("${size}/${rep}: ${jid}")
    echo "submitted ${size}/${rep} as job ${jid}"
  done
done

echo
echo "=== submission summary ==="
echo "submitted: ${#submitted[@]}"
for s in "${submitted[@]}"; do echo "  ${s}"; done
if (( ${#errors[@]} > 0 )); then
  echo "errors: ${#errors[@]}"
  for e in "${errors[@]}"; do echo "  ${e}"; done
fi
