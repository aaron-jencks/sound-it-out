#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
TRAIN_SLURM="${REPO_DIR}/multilingual_8lang_unified_100k/slurm_train_xlsum_sft.slurm"
SAMPLE_SLURM="${REPO_DIR}/multilingual_8lang_unified_100k/slurm_sample_xlsum_sft_checkpoints.slurm"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"
LOG_DIR="${EXP_BASE}/logs"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXP_BASE}/xlsum_sft/pilot_runs}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${EXP_BASE}/hf_cache}"
DRY_RUN="${DRY_RUN:-0}"

REPS=(text ipa_stripped romanized)
SIZE="${SIZE:-medium}"
MIX_MODE="${MIX_MODE:-balanced}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-256}"
EPOCHS="${EPOCHS:-0.5}"
MAX_STEPS="${MAX_STEPS:-600}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-1e-5}"
WD="${WD:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-50}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
TRAIN_TIME="${TRAIN_TIME:-02:30:00}"
SAMPLE_TIME="${SAMPLE_TIME:-01:00:00}"
SAMPLE_LANGUAGES="${SAMPLE_LANGUAGES:-english+spanish+hindi+russian+tamil+urdu}"
SAMPLE_SPLIT="${SAMPLE_SPLIT:-validation}"
SAMPLES_PER_LANG="${SAMPLES_PER_LANG:-1}"
MAX_EVAL_SAMPLES_PER_LANG="${MAX_EVAL_SAMPLES_PER_LANG:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
NUM_BEAMS="${NUM_BEAMS:-4}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-3}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

format_epoch_tag() {
  local epoch="$1"
  echo "${epoch}" | tr '.' 'p'
}

packed_data_dir_for() {
  local rep="$1"
  if [[ "${rep}" == "text" ]]; then
    echo "${EXP_BASE}/xlsum_sft/packs/text_balanced_ctx${CONTEXT_LEN}_tgt${TARGET_MAX_TOKENS}"
  else
    echo "${EXP_BASE}/xlsum_sft/packs/${rep}_${MIX_MODE}_ctx${CONTEXT_LEN}_tgt${TARGET_MAX_TOKENS}_${SIZE}_pilot"
  fi
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

submit_cmd() {
  local dependency="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "$@"
  else
    if [[ -n "${dependency}" ]]; then
      "$@" --dependency="afterok:${dependency}"
    else
      "$@"
    fi
  fi
}

epoch_tag="$(format_epoch_tag "${EPOCHS}")"

for rep in "${REPS[@]}"; do
  run_name="${SIZE}_${rep}_pilot_lr${LR}_wd${WD}_wr${WARMUP_RATIO}_e${epoch_tag}_ms${MAX_STEPS}_s${SEED}"
  run_dir="${OUTPUT_ROOT}/${rep}/${SIZE}/${run_name}"
  packed_data_dir="$(packed_data_dir_for "${rep}")"

  train_export=(
    "ALL"
    "EXP_BASE=${EXP_BASE}"
    "REP=${rep}"
    "SIZE=${SIZE}"
    "MIX_MODE=${MIX_MODE}"
    "CONTEXT_LEN=${CONTEXT_LEN}"
    "TARGET_MAX_TOKENS=${TARGET_MAX_TOKENS}"
    "EPOCHS=${EPOCHS}"
    "MAX_STEPS=${MAX_STEPS}"
    "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    "GRAD_ACCUM=${GRAD_ACCUM}"
    "LR=${LR}"
    "WD=${WD}"
    "WARMUP_RATIO=${WARMUP_RATIO}"
    "EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS}"
    "SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"
    "MIXED_PRECISION=${MIXED_PRECISION}"
    "SEED=${SEED}"
    "DATASET_REPO=${DATASET_REPO}"
    "RUN_NAME=${run_name}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "PACKED_DATA_DIR=${packed_data_dir}"
    "DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
  )
  train_export_string="$(IFS=,; echo "${train_export[*]}")"

  sample_export=(
    "ALL"
    "EXP_BASE=${EXP_BASE}"
    "RUN_DIR=${run_dir}"
    "REP=${rep}"
    "SIZE=${SIZE}"
    "DATASET_REPO=${DATASET_REPO}"
    "DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
    "SPLIT=${SAMPLE_SPLIT}"
    "LANGUAGES=${SAMPLE_LANGUAGES}"
    "SAMPLES_PER_LANG=${SAMPLES_PER_LANG}"
    "MAX_EVAL_SAMPLES_PER_LANG=${MAX_EVAL_SAMPLES_PER_LANG}"
    "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
    "NUM_BEAMS=${NUM_BEAMS}"
    "REPETITION_PENALTY=${REPETITION_PENALTY}"
    "NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE}"
    "LENGTH_PENALTY=${LENGTH_PENALTY}"
    "OUT_DIR=${run_dir}/sample_checks"
  )
  sample_export_string="$(IFS=,; echo "${sample_export[*]}")"

  train_job_name="xlsum-pilot-${rep}"
  sample_job_name="xlsum-pilot-sample-${rep}"

  echo "Submitting ${train_job_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print \
      sbatch \
      --job-name="${train_job_name}" \
      --time="${TRAIN_TIME}" \
      --gpus-per-node="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${LOG_DIR}/${train_job_name}-%j.out" \
      --error="${LOG_DIR}/${train_job_name}-%j.err" \
      --export="${train_export_string}" \
      "${TRAIN_SLURM}"
    echo "Submitting ${sample_job_name}"
    run_or_print \
      sbatch \
      --job-name="${sample_job_name}" \
      --time="${SAMPLE_TIME}" \
      --gpus-per-node=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${LOG_DIR}/${sample_job_name}-%j.out" \
      --error="${LOG_DIR}/${sample_job_name}-%j.err" \
      --export="${sample_export_string}" \
      "${SAMPLE_SLURM}"
    continue
  fi

  train_job_id="$(
    sbatch --parsable \
      --job-name="${train_job_name}" \
      --time="${TRAIN_TIME}" \
      --gpus-per-node="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${LOG_DIR}/${train_job_name}-%j.out" \
      --error="${LOG_DIR}/${train_job_name}-%j.err" \
      --export="${train_export_string}" \
      "${TRAIN_SLURM}"
  )"
  echo "  train job id: ${train_job_id}"

  echo "Submitting ${sample_job_name}"
  sample_job_id="$(
    sbatch --parsable \
      --dependency="afterok:${train_job_id}" \
      --job-name="${sample_job_name}" \
      --time="${SAMPLE_TIME}" \
      --gpus-per-node=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${LOG_DIR}/${sample_job_name}-%j.out" \
      --error="${LOG_DIR}/${sample_job_name}-%j.err" \
      --export="${sample_export_string}" \
      "${SAMPLE_SLURM}"
  )"
  echo "  sample job id: ${sample_job_id}"
done
