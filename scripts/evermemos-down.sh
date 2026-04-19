#!/usr/bin/env bash
# Stop EverMemOS infrastructure and remove containers (volumes preserved).

set -euo pipefail

WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR_HOST="${REPO_ROOT}/external/everos/methods/evermemos"
COMPOSE_DIR_WSL=$(echo "${COMPOSE_DIR_HOST}" | sed -E 's|^([A-Za-z]):/|/mnt/\L\1/|')

echo "[evermemos-down] stopping services"
wsl.exe -d "${WSL_DISTRO}" -- bash -lc "cd '${COMPOSE_DIR_WSL}' && docker compose down"
echo "[evermemos-down] done (volumes preserved — use docker compose down -v to wipe data)"
