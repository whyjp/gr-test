#!/usr/bin/env bash
# Tail logs from EverMemOS containers. Pass service name as $1 to limit scope
# (e.g. ./evermemos-logs.sh milvus-standalone).

set -euo pipefail

WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR_HOST="${REPO_ROOT}/external/everos/methods/evermemos"
COMPOSE_DIR_WSL=$(echo "${COMPOSE_DIR_HOST}" | sed -E 's|^([A-Za-z]):/|/mnt/\L\1/|')

SERVICE="${1:-}"
TAIL_N="${TAIL_N:-200}"

if [ -n "${SERVICE}" ]; then
  echo "[evermemos-logs] tailing ${SERVICE} (last ${TAIL_N})"
  wsl.exe -d "${WSL_DISTRO}" -- bash -lc \
    "cd '${COMPOSE_DIR_WSL}' && docker compose logs --tail=${TAIL_N} -f '${SERVICE}'"
else
  echo "[evermemos-logs] tailing all services (last ${TAIL_N})"
  wsl.exe -d "${WSL_DISTRO}" -- bash -lc \
    "cd '${COMPOSE_DIR_WSL}' && docker compose logs --tail=${TAIL_N} -f"
fi
