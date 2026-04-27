#!/usr/bin/env bash
# Submit XLSum SFT (per-language best-checkpoint) jobs.
#
# Default: 4 sizes × 3 reps = 12 train jobs, each with a dependent eval job.
# Override SIZES / REPS to submit a subset.
#
#     bash submit_xlsum_perlang.sh                   # all 12 train + eval pairs
#     SIZES="small medium" bash submit_xlsum_perlang.sh
#     REPS="text" bash submit_xlsum_perlang.sh
#     DRY_RUN=1 bash submit_xlsum_perlang.sh         # print, don't submit

set -euo pipefail

REPO_DIR="${REPO_DIR:-/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training}"
BUNDLE_DIR="${BUNDLE_DIR:-${REPO_DIR}/bundle/finetuning/xlsum}"
TRAIN_SLURM="${BUNDLE_DIR}/slurm/slurm_train.slurm"
EVAL_SLURM="${BUNDLE_DIR}/slurm/slurm_eval.slurm"

EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"
LOG_DIR="${LOG_DIR:-${EXP_BASE}/logs}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${EXP_BASE}/hf_cache}"
DRY_RUN="${DRY_RUN:-0}"

# Job matrix
SIZES_DEFAULT="small medium large_standard xl"
REPS_DEFAULT="text ipa_stripped romanized"
SIZES="${SIZES:-$SIZES_DEFAULT}"
REPS="${REPS:-$REPS_DEFAULT}"

# Training defaults (override via env)
ACCOUNT="${ACCOUNT:-PAS2836}"
PARTITION="${PARTITION:-gpu}"
MIX_MODE="${MIX_MODE:-balanced}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-256}"
EPOCHS="${EPOCHS:-3.0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WD="${WD:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500}"
VAL_SAMPLES_PER_LANG="${VAL_SAMPLES_PER_LANG:-0}"      # 0 = full val pass
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

# Eval settings
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"              # 0 = full eval

mkdir -p "${LOG_DIR}"

format_epoch_tag() { echo "${1}" | tr '.' 'p'; }

packed_data_dir_for() {
  local rep="$1" size="$2"
  echo "${EXP_BASE}/xlsum_sft/packs/${rep}_${MIX_MODE}_ctx${CONTEXT_LEN}_tgt${TARGET_MAX_TOKENS}_${size}_jobpack"
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "$@"; printf '\n'
  else "$@"
  fi
}

epoch_tag="$(format_epoch_tag "${EPOCHS}")"
output_root="${EXP_BASE}/xlsum_sft/perlang_runs"

submitted=()

for size in ${SIZES}; do
  case "${size}" in
    small)            train_time="2-00:00:00"; eval_time="10:00:00";    lr_default="5e-6" ;;
    medium)           train_time="2-00:00:00"; eval_time="1-00:00:00";  lr_default="3e-6" ;;
    large_standard)   train_time="3-00:00:00"; eval_time="1-00:00:00";  lr_default="1e-6" ;;
    xl)               train_time="4-00:00:00"; eval_time="1-12:00:00";  lr_default="5e-7" ;;
    *) echo "ERROR: unsupported size '${size}'" >&2; exit 1 ;;
  esac

  # Per-size LR override: LR_SMALL / LR_MEDIUM / LR_LARGE_STANDARD / LR_XL
  size_upper="$(echo "${size}" | tr '[:lower:]' '[:upper:]')"
  lr_var="LR_${size_upper}"
  lr="${!lr_var:-${LR:-$lr_default}}"

  for rep in ${REPS}; do
    job_name="xlsum-perlang-${size}-${rep}"
    eval_job_name="xlsum-eval-${size}-${rep}"
    packed_data_dir="$(packed_data_dir_for "${rep}" "${size}")"
    run_name="${size}_${rep}_lr${lr}_wd${WD}_wr${WARMUP_RATIO}_e${epoch_tag}_s${SEED}"
    ckpt_dir="${output_root}/${rep}/${size}/${run_name}"

    train_out="${LOG_DIR}/${job_name}-%j.out"
    train_err="${LOG_DIR}/${job_name}-%j.err"
    eval_out="${LOG_DIR}/${eval_job_name}-%j.out"
    eval_err="${LOG_DIR}/${eval_job_name}-%j.err"

    train_export="ALL"
    train_export+=",REPO_DIR=${REPO_DIR}"
    train_export+=",BUNDLE_DIR=${BUNDLE_DIR}"
    train_export+=",EXP_BASE=${EXP_BASE}"
    train_export+=",REP=${rep}"
    train_export+=",SIZE=${size}"
    train_export+=",MIX_MODE=${MIX_MODE}"
    train_export+=",CONTEXT_LEN=${CONTEXT_LEN}"
    train_export+=",TARGET_MAX_TOKENS=${TARGET_MAX_TOKENS}"
    train_export+=",EPOCHS=${EPOCHS}"
    train_export+=",PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    train_export+=",GRAD_ACCUM=${GRAD_ACCUM}"
    train_export+=",LR=${lr}"
    train_export+=",WD=${WD}"
    train_export+=",WARMUP_RATIO=${WARMUP_RATIO}"
    train_export+=",EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS}"
    train_export+=",VAL_SAMPLES_PER_LANG=${VAL_SAMPLES_PER_LANG}"
    train_export+=",MIXED_PRECISION=${MIXED_PRECISION}"
    train_export+=",SEED=${SEED}"
    train_export+=",DATASET_REPO=${DATASET_REPO}"
    train_export+=",DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
    train_export+=",RUN_NAME=${run_name}"
    train_export+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
    train_export+=",PACKED_DATA_DIR=${packed_data_dir}"
    train_export+=",OUTPUT_ROOT=${output_root}"

    eval_export="ALL"
    eval_export+=",REPO_DIR=${REPO_DIR}"
    eval_export+=",BUNDLE_DIR=${BUNDLE_DIR}"
    eval_export+=",EXP_BASE=${EXP_BASE}"
    eval_export+=",CKPT_DIR=${ckpt_dir}"
    eval_export+=",REP=${rep}"
    eval_export+=",SIZE=${size}"
    eval_export+=",SPLIT=${EVAL_SPLIT}"
    eval_export+=",DATASET_REPO=${DATASET_REPO}"
    eval_export+=",MAX_EVAL_SAMPLES_PER_LANG=${EVAL_MAX_SAMPLES}"
    eval_export+=",OUT_DIR=${EXP_BASE}/xlsum_sft/eval_perlang/${rep}/${size}"

    echo "Submitting train: ${job_name} (lr=${lr}, time=${train_time})"
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_or_print sbatch \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --time="${train_time}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --output="${train_out}" --error="${train_err}" \
        --export="${train_export}" \
        "${TRAIN_SLURM}"
      echo "[DRY] would chain eval ${eval_job_name}"
      continue
    fi

    train_jid=$(sbatch --parsable \
      --account="${ACCOUNT}" --partition="${PARTITION}" \
      --job-name="${job_name}" --time="${train_time}" \
      --gpus-per-node="${GPUS_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${train_out}" --error="${train_err}" \
      --export="${train_export}" \
      "${TRAIN_SLURM}")
    echo "  -> train ${train_jid}"

    eval_jid=$(sbatch --parsable \
      --account="${ACCOUNT}" --partition="${PARTITION}" \
      --job-name="${eval_job_name}" --time="${eval_time}" \
      --gpus-per-node=1 --cpus-per-task=8 \
      --output="${eval_out}" --error="${eval_err}" \
      --export="${eval_export}" \
      --dependency="afterok:${train_jid}" \
      "${EVAL_SLURM}")
    echo "  -> eval  ${eval_jid} (depends on ${train_jid})"

    submitted+=("${size}/${rep}: train=${train_jid} eval=${eval_jid}")
  done
done

echo
echo "=== submitted (${#submitted[@]} pairs) ==="
for s in "${submitted[@]}"; do echo "  ${s}"; done
