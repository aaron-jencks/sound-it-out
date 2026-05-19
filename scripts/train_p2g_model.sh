#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

MODE="${1:-ipa}"
case "${MODE}" in
  ipa)
    MODE_CONFIG="transcription/p2g/config/ipa_train_eval.json"
    ;;
  roman)
    MODE_CONFIG="transcription/p2g/config/roman_train_eval.json"
    ;;
  *)
    echo "usage: $(basename "$0") [ipa|roman]" >&2
    exit 1
    ;;
esac

CONFIG_ARGS=("${MODE_CONFIG}")
LOCAL_CONFIG="transcription/p2g/config/local_train.json"
if [[ -f "${SRC_ROOT}/${LOCAL_CONFIG}" ]]; then
  CONFIG_ARGS+=("${LOCAL_CONFIG}")
fi

cd "${SRC_ROOT}"
python -m transcription.p2g.train_t5 \
  "transcription/p2g/config/default_train_eval.json" \
  "${CONFIG_ARGS[@]}"
