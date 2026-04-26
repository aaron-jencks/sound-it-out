#!/bin/bash
#SBATCH --job-name=xsc-zs
#SBATCH --account=PAS2836
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=xsc-zs-%x-%j.out
#SBATCH --error=xsc-zs-%x-%j.err

# Parametrized via --export when calling sbatch:
#   sbatch --job-name=xsc-zs-en-text --export=ALL,LANG=en,REPR=text submit_zeroshot.sh
#
# Or called directly from submit_all_zeroshot.sh.
#
# NOTE: XStoryCloze evaluation split is named "eval", not "test".
# The loader maps --split eval -> HF split "eval".

set -euo pipefail

CALIBRATE="${CALIBRATE:-none}"

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "LANG: ${LANG}"
echo "REPR: ${REPR}"
echo "CALIBRATE: ${CALIBRATE}  (no-op for MCQ task family)"
echo "Start: $(date)"

REPO_DIR="/fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training"
SCRIPT="${REPO_DIR}/generative-classification-ipa-gpt-calibrate/icl_eval.py"

/usr/bin/python "$SCRIPT" \
    --dataset xstorycloze \
    --language "${LANG}" \
    --representation "${REPR}" \
    --k 0 \
    --subsample 500 \
    --split eval \
    --seed 42 \
    --calibrate "${CALIBRATE}"

echo "Finished: $(date)"
