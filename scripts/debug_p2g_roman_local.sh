#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"
UROMAN_PATH="/home/aaron/Documents/workspace/research/kumar/uroman-1.2.8/bin/uroman.pl"

if [[ ! -f "${UROMAN_PATH}" ]]; then
  echo "uroman.pl not found at ${UROMAN_PATH}" >&2
  exit 1
fi

cd "${SRC_ROOT}"
python -m transcription.p2g.dataset \
  --debug \
  --default-config transcription/p2g/config/default_pre.json \
  transcription/p2g/config/roman_pre.json \
  transcription/p2g/config/roman_local_debug.json \
  "$@"
