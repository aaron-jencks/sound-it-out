#!/bin/bash
#SBATCH --job-name=xlsum-perlang
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Per-language XLSum SFT training. Pairs with bundle/finetuning/xlsum/train_xlsum_perlang.py.
# SIGUSR1 arrives 180s before walltime → script saves last.pt and exits gracefully.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training}"
BUNDLE_DIR="${BUNDLE_DIR:-${REPO_DIR}/bundle/finetuning/xlsum}"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"

REP="${REP:?REP not set (text|ipa_stripped|romanized)}"
SIZE="${SIZE:?SIZE not set (small|medium|large_standard|xl)}"
MIX_MODE="${MIX_MODE:-balanced}"

CONTEXT_LEN="${CONTEXT_LEN:-2048}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-256}"
EPOCHS="${EPOCHS:-3.0}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_SAMPLE_FRAC="${TRAIN_SAMPLE_FRAC:-1.0}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-3e-6}"
WD="${WD:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-800}"
VAL_SAMPLES_PER_LANG="${VAL_SAMPLES_PER_LANG:-0}"   # 0 = full val pass
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"
PACKED_DATA_DIR="${PACKED_DATA_DIR:-}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
ENV_NAME="${ENV_NAME:-ipagpt_training}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

RUN_NAME="${RUN_NAME:-${SIZE}_${REP}_xlsum_sft_perlang}"

cd "${REPO_DIR}"
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="^lo,docker0"
export TORCHINDUCTOR_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(hostname)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

CMD=(
  python -m torch.distributed.run
  --nproc-per-node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  "${BUNDLE_DIR}/train_xlsum_perlang.py"
  --exp_base "${EXP_BASE}"
  --dataset_repo "${DATASET_REPO}"
  --representation "${REP}"
  --size "${SIZE}"
  --mix_mode "${MIX_MODE}"
  --context_len "${CONTEXT_LEN}"
  --target_max_tokens "${TARGET_MAX_TOKENS}"
  --num_train_epochs "${EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --train_sample_frac "${TRAIN_SAMPLE_FRAC}"
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LR}"
  --weight_decay "${WD}"
  --warmup_ratio "${WARMUP_RATIO}"
  --val_samples_per_lang "${VAL_SAMPLES_PER_LANG}"
  --eval_every_steps "${EVAL_EVERY_STEPS}"
  --save_every_steps "${SAVE_EVERY_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --seed "${SEED}"
  --run_name "${RUN_NAME}"
  --build_packs_if_missing
)

[[ -n "${PACKED_DATA_DIR}" ]] && CMD+=(--packed_data_dir "${PACKED_DATA_DIR}")
[[ -n "${DATASET_CACHE_DIR}" ]] && CMD+=(--dataset_cache_dir "${DATASET_CACHE_DIR}")
[[ -n "${OUTPUT_ROOT}" ]] && CMD+=(--output_root "${OUTPUT_ROOT}")
[[ -n "${RESUME_CKPT}" ]] && CMD+=(--resume_ckpt "${RESUME_CKPT}")
[[ "${SKIP_FINAL_EVAL}" == "1" ]] && CMD+=(--skip_final_eval)

# Forward SIGUSR1/SIGTERM from slurm to the python process.
TRAIN_PID=""
forward_signal() { local sig="$1"; [[ -n "${TRAIN_PID}" ]] && kill "-${sig}" "${TRAIN_PID}" 2>/dev/null || true; }
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

"${CMD[@]}" &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
status=$?
set -e
trap - USR1 TERM
exit "${status}"
