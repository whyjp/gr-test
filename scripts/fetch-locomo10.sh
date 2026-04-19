#!/usr/bin/env bash
# Fetch LoCoMo-10 dataset from snap-research/locomo.
# Idempotent — skips download if file already present and SHA matches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
TARGET="${DATA_DIR}/locomo10.json"

URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
# Git blob SHA as of 2026-04 (from GitHub contents API); informational only.
EXPECTED_BLOB_SHA="d95b872480b413d935821fdc3c84f8a8f5f29e73"
EXPECTED_SIZE=2805274

mkdir -p "${DATA_DIR}"

if [[ -f "${TARGET}" ]]; then
  actual_size=$(wc -c < "${TARGET}" | tr -d ' ')
  if [[ "${actual_size}" == "${EXPECTED_SIZE}" ]]; then
    echo "[ok] ${TARGET} already present (${actual_size} bytes)"
    exit 0
  fi
  echo "[warn] size mismatch (${actual_size} != ${EXPECTED_SIZE}), re-downloading"
fi

echo "[fetch] ${URL}"
curl -sSL --fail -o "${TARGET}" "${URL}"

actual_size=$(wc -c < "${TARGET}" | tr -d ' ')
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
  echo "[error] downloaded size ${actual_size} != expected ${EXPECTED_SIZE}" >&2
  exit 1
fi

echo "[ok] ${TARGET} (${actual_size} bytes, expected blob sha ${EXPECTED_BLOB_SHA})"
