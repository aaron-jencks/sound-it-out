#!/bin/bash
# Submit zero-shot PAWS-X eval jobs for all language × representation combinations.
#
# Supported languages: en, es  (fr, zh not in 8-lang training set)
# Supported representations: text, romanized, ipa
#
# Usage:
#   cd generative-classification-ipa-gpt-calibrate/pawsx
#   bash submit_all_zeroshot.sh
#
# Optional: filter to a single language or representation:
#   bash submit_all_zeroshot.sh en          # all representations for en only
#   bash submit_all_zeroshot.sh en text     # single job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER_LANG="${1:-all}"
FILTER_REPR="${2:-all}"

# Calibration mode comes from env (default "none" — backward compatible).
# Set CALIBRATE=cc to opt into Zhao 2021 contextual calibration. Job names
# and logs get a "-cc" suffix so calibrated and raw runs don't collide.
CALIBRATE="${CALIBRATE:-none}"
SUFFIX=""
[[ "$CALIBRATE" != "none" ]] && SUFFIX="-${CALIBRATE}"

LANGS=(en es)
REPRS=(text romanized ipa)

for lang in "${LANGS[@]}"; do
    for repr in "${REPRS[@]}"; do
        [[ "$FILTER_LANG" != "all" && "$FILTER_LANG" != "$lang" ]] && continue
        [[ "$FILTER_REPR" != "all" && "$FILTER_REPR" != "$repr" ]] && continue

        job_name="pawsx-zs-${lang}-${repr}${SUFFIX}"
        out_file="${SCRIPT_DIR}/logs/${job_name}-%j.out"
        err_file="${SCRIPT_DIR}/logs/${job_name}-%j.err"

        mkdir -p "${SCRIPT_DIR}/logs"

        jid=$(sbatch \
            --job-name="${job_name}" \
            --output="${out_file}" \
            --error="${err_file}" \
            --export=ALL,LANG="${lang}",REPR="${repr}",CALIBRATE="${CALIBRATE}" \
            "${SCRIPT_DIR}/submit_zeroshot.sh" \
            --parsable)

        echo "Submitted ${job_name}: job ${jid}"
    done
done
