#!/bin/bash
#SBATCH --job-name=xlsum-audit-ckpts
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"

RUN_DIR="${RUN_DIR:?RUN_DIR not set}"
REP="${REP:?REP not set (text|ipa_stripped|romanized)}"
SIZE="${SIZE:?SIZE not set (small|medium|large)}"
DATASET_REPO="${DATASET_REPO:-mugezhang/xlsum_6lang_multirepr}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"
SPLIT="${SPLIT:-validation}"
LANGUAGES="${LANGUAGES:-english+spanish+hindi+russian+tamil+urdu}"
SAMPLES_PER_LANG="${SAMPLES_PER_LANG:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
NUM_BEAMS="${NUM_BEAMS:-4}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-3}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.0}"
MAX_EVAL_SAMPLES_PER_LANG="${MAX_EVAL_SAMPLES_PER_LANG:-8}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/sample_checks}"
LANGUAGES_CSV="${LANGUAGES//+/,}"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ipagpt_training

mapfile -t ckpts < <(
  find "${RUN_DIR}" -maxdepth 1 -type f \
    \( -name 'best.pt' -o -name 'last.pt' -o -name 'step_*.pt' \) | sort
)

if [[ "${#ckpts[@]}" -eq 0 ]]; then
  echo "No checkpoints found in ${RUN_DIR}" >&2
  exit 1
fi

manifest="${OUT_DIR}/manifest.tsv"
printf "checkpoint\tmetrics_json\tsamples_jsonl\n" > "${manifest}"

for ckpt in "${ckpts[@]}"; do
  base_name="$(basename "${ckpt}" .pt)"
  metrics_json="${OUT_DIR}/${base_name}_${SPLIT}_metrics.json"
  metrics_csv="${OUT_DIR}/${base_name}_${SPLIT}_metrics.csv"
  out_jsonl="${OUT_DIR}/${base_name}_${SPLIT}_samples.jsonl"
  echo "Evaluating ${ckpt} -> ${metrics_json}"
  python -u evaluate_xlsum_generation.py \
    --model_ckpt "${ckpt}" \
    --representation "${REP}" \
    --size "${SIZE}" \
    --dataset_repo "${DATASET_REPO}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --split "${SPLIT}" \
    --exp_base "${EXP_BASE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_beams "${NUM_BEAMS}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}" \
    --length_penalty "${LENGTH_PENALTY}" \
    --max_eval_samples_per_lang "${MAX_EVAL_SAMPLES_PER_LANG}" \
    --device cuda \
    --output_json "${metrics_json}" \
    --output_csv "${metrics_csv}"
  echo "Sampling ${ckpt} -> ${out_jsonl}"
  python -u bundle/finetuning/xlsum/sample_xlsum_generations.py \
    --model_ckpt "${ckpt}" \
    --representation "${REP}" \
    --size "${SIZE}" \
    --dataset_repo "${DATASET_REPO}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --split "${SPLIT}" \
    --exp_base "${EXP_BASE}" \
    --languages "${LANGUAGES_CSV}" \
    --samples_per_lang "${SAMPLES_PER_LANG}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_beams "${NUM_BEAMS}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}" \
    --length_penalty "${LENGTH_PENALTY}" \
    --device cuda \
    --output_jsonl "${out_jsonl}"
  printf "%s\t%s\t%s\n" "${ckpt}" "${metrics_json}" "${out_jsonl}" >> "${manifest}"
done

echo "Saved manifest to ${manifest}"
