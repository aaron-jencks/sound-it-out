#!/bin/bash
#SBATCH --job-name=xnli-zs
#SBATCH --account=PAS2836
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=xnli-zs-%x-%j.out
#SBATCH --error=xnli-zs-%x-%j.err

# Parametrized via --export when calling sbatch:
#   sbatch --job-name=xnli-zs-en-text --export=ALL,LANG=en,REPR=text submit_zeroshot.sh
#
# Or called directly from submit_all_zeroshot.sh.

set -euo pipefail

CALIBRATE="${CALIBRATE:-none}"

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "LANG: ${LANG}"
echo "REPR: ${REPR}"
echo "CALIBRATE: ${CALIBRATE}"
echo "Start: $(date)"

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
SCRIPT="${REPO_DIR}/generative-classification-ipa-gpt-calibrate/icl_eval.py"

/usr/bin/python "$SCRIPT" \
    --dataset xnli \
    --language "${LANG}" \
    --representation "${REPR}" \
    --k 0 \
    --subsample 500 \
    --split validation \
    --seed 42 \
    --calibrate "${CALIBRATE}"

echo "Finished: $(date)"