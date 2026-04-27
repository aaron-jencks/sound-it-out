#!/bin/bash
#SBATCH --job-name=xlsum-hpo-s1
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"

REP="${REP:?REP not set (text|ipa_stripped|romanized)}"
SIZE="${SIZE:-medium}"
MIX_MODE="${MIX_MODE:-balanced}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"

cd "${REPO_DIR}"
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ipagpt_training

python bundle/finetuning/xlsum/hpo_xlsum_grid.py \
  --stage 1 \
  --representation "${REP}" \
  --size "${SIZE}" \
  --exp_base "${EXP_BASE}" \
  --dataset_repo "${DATASET_REPO}" \
  --mix_mode "${MIX_MODE}" \
  --context_len "${CONTEXT_LEN:-2048}" \
  --target_max_tokens "${TARGET_MAX_TOKENS:-256}" \
  --num_beams "${NUM_BEAMS:-4}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-16}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}"
