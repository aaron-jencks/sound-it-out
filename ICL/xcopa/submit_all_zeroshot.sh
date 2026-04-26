#!/bin/bash
# Submit zero-shot XCopa eval jobs for all representation variants.
#
# Supported languages: ta (Tamil only — zh not in 8-lang training set)
# Supported representations: text, romanized, ipa
#
# Usage:
#   cd generative-classification-ipa-gpt-calibrate/xcopa
#   bash submit_all_zeroshot.sh
#
# Optional: filter to a single representation:
#   bash submit_all_zeroshot.sh text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER_REPR="${1:-all}"

# CALIBRATE is forwarded for consistency with label-word task scripts but
# is a no-op for MCQ (no shared label tokens; CC priors aren't well-defined).
CALIBRATE="${CALIBRATE:-none}"
SUFFIX=""
[[ "$CALIBRATE" != "none" ]] && SUFFIX="-${CALIBRATE}"

LANG="ta"
REPRS=(text romanized ipa)

for repr in "${REPRS[@]}"; do
    [[ "$FILTER_REPR" != "all" && "$FILTER_REPR" != "$repr" ]] && continue

    job_name="xcopa-zs-${LANG}-${repr}${SUFFIX}"
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
