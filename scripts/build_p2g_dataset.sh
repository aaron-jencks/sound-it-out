#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

MODE="${1:-ipa}"
case "${MODE}" in
  ipa)
    MODE_CONFIG="transcription/p2g/config/ipa_pre.json"
    ;;
  roman)
    MODE_CONFIG="transcription/p2g/config/roman_pre.json"
    ;;
  debug)
    MODE_CONFIG="transcription/p2g/config/debug_ipa_pre.json"
    ;;
  *)
    echo "usage: $(basename "$0") [ipa|roman|debug]" >&2
    exit 1
    ;;
esac

CONFIG_ARGS=("${MODE_CONFIG}")
LOCAL_CONFIG="transcription/p2g/config/local.json"
if [[ -f "${SRC_ROOT}/${LOCAL_CONFIG}" ]]; then
  CONFIG_ARGS+=("${LOCAL_CONFIG}")
fi

cd "${SRC_ROOT}"
python -m transcription.p2g.dataset \
  "transcription/p2g/config/default_pre.json" \
  "${CONFIG_ARGS[@]}"
