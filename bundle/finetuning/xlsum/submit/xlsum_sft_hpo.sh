#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
SLURM_DIR="${REPO_DIR}/multilingual_8lang_unified_100k"
EXP_BASE="${EXP_BASE:-/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k}"

MODE="${1:-all}" # stage1 | stage2 | all

REPS=(text ipa_stripped romanized)
SIZES=(small medium large)

submit_stage1() {
  for rep in "${REPS[@]}"; do
    job_name="xlsum-s1-${rep}"
    echo "Submitting stage1 ${job_name}"
    sbatch \
      --job-name="${job_name}" \
      --export=ALL,EXP_BASE="${EXP_BASE}",REP="${rep}",SIZE=medium,MIX_MODE=balanced \
      "${SLURM_DIR}/slurm_hpo_xlsum_stage1.slurm"
  done
}

submit_stage2() {
  for size in "${SIZES[@]}"; do
    for rep in "${REPS[@]}"; do
      s1_best="${EXP_BASE}/xlsum_sft/hpo/stage1/medium_${rep}/best_config.json"
      job_name="xlsum-s2-${size}-${rep}"
      echo "Submitting stage2 ${job_name}"
      sbatch \
        --job-name="${job_name}" \
        --export=ALL,EXP_BASE="${EXP_BASE}",REP="${rep}",SIZE="${size}",MIX_MODE=balanced,TRIALS=8,STAGE1_BEST_JSON="${s1_best}" \
        "${SLURM_DIR}/slurm_hpo_xlsum_stage2.slurm"
    done
  done
}

case "${MODE}" in
  stage1)
    submit_stage1
    ;;
  stage2)
    submit_stage2
    ;;
  all)
    submit_stage1
    submit_stage2
    ;;
  *)
    echo "Usage: $0 [stage1|stage2|all]"
    exit 1
    ;;
esac
