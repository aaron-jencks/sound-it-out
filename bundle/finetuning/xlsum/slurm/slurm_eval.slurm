#!/bin/bash
#SBATCH --job-name=xlsum-eval
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Run evaluate_xlsum_perlang.py against a directory of best_{lang}.pt files.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training}"
BUNDLE_DIR="${BUNDLE_DIR:-${REPO_DIR}/bundle/finetuning/xlsum}"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"

CKPT_DIR="${CKPT_DIR:?CKPT_DIR not set (directory containing best_{lang}.pt)}"
REP="${REP:?REP not set (text|ipa_stripped|romanized)}"
SIZE="${SIZE:?SIZE not set (small|medium|large_standard|xl)}"
SPLIT="${SPLIT:-test}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"

CONTEXT_LEN="${CONTEXT_LEN:-2048}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-256}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_EVAL_SAMPLES_PER_LANG="${MAX_EVAL_SAMPLES_PER_LANG:-0}"
ENV_NAME="${ENV_NAME:-ipagpt_training}"

OUT_DIR="${OUT_DIR:-${EXP_BASE}/xlsum_sft/eval_perlang/${REP}/${SIZE}}"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/metrics_${SPLIT}_${SLURM_JOB_ID}.json}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/metrics_${SPLIT}_${SLURM_JOB_ID}.csv}"
OUT_PREDS_DIR="${OUT_PREDS_DIR:-${OUT_DIR}/preds_${SPLIT}_${SLURM_JOB_ID}}"

cd "${REPO_DIR}"
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python "${BUNDLE_DIR}/evaluate_xlsum_perlang.py" \
  --ckpt_dir "${CKPT_DIR}" \
  --representation "${REP}" \
  --size "${SIZE}" \
  --dataset_repo "${DATASET_REPO}" \
  --split "${SPLIT}" \
  --exp_base "${EXP_BASE}" \
  --context_len "${CONTEXT_LEN}" \
  --target_max_tokens "${TARGET_MAX_TOKENS}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_eval_samples_per_lang "${MAX_EVAL_SAMPLES_PER_LANG}" \
  --output_json "${OUT_JSON}" \
  --output_csv "${OUT_CSV}" \
  --output_preds_dir "${OUT_PREDS_DIR}"
