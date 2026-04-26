#!/bin/bash
# Submit zero-shot SST-2 eval jobs for all representation variants.
#
# Usage:
#   cd generative-classification-ipa-gpt-calibrate/sst2
#   bash submit_all_zeroshot.sh
#
# Optional: filter to a single representation:
#   bash submit_all_zeroshot.sh text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER_REPR="${1:-all}"

# Calibration mode comes from env (default "none" — backward compatible).
# Set CALIBRATE=cc to opt into Zhao 2021 contextual calibration. Job names
# and logs get a "-cc" suffix so calibrated and raw runs don't collide.
CALIBRATE="${CALIBRATE:-none}"
SUFFIX=""
[[ "$CALIBRATE" != "none" ]] && SUFFIX="-${CALIBRATE}"

LANG="en"
REPRS=(text romanized ipa)

for repr in "${REPRS[@]}"; do
    [[ "$FILTER_REPR" != "all" && "$FILTER_REPR" != "$repr" ]] && continue

    job_name="sst2-zs-${LANG}-${repr}${SUFFIX}"
    out_file="${SCRIPT_DIR}/logs/${job_name}-%j.out"
    err_file="${SCRIPT_DIR}/logs/${job_name}-%j.err"

    mkdir -p "${SCRIPT_DIR}/logs"

    jid=$(sbatch \
        --job-name="${job_name}" \
        --output="${out_file}" \
        --error="${err_file}" \
        --export=ALL,LANG="${LANG}",REPR="${repr}",CALIBRATE="${CALIBRATE}" \
        "${SCRIPT_DIR}/submit_zeroshot.sh" \
        --parsable)

    echo "Submitted ${job_name}: job ${jid}"
done