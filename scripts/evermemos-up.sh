#!/usr/bin/env bash
# Start EverMemOS infrastructure (MongoDB + ES + Milvus + Redis) via WSL docker.
# Idempotent: re-running is safe; waits for readiness before returning.

set -euo pipefail

WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR_HOST="${REPO_ROOT}/external/everos/methods/evermemos"
COMPOSE_DIR_WSL=$(echo "${COMPOSE_DIR_HOST}" | sed -E 's|^([A-Za-z]):/|/mnt/\L\1/|')
READINESS_TIMEOUT="${READINESS_TIMEOUT:-120}"

echo "[evermemos-up] WSL distro: ${WSL_DISTRO}"
echo "[evermemos-up] compose dir: ${COMPOSE_DIR_WSL}"

wsl.exe -d "${WSL_DISTRO}" -- bash -lc "cd '${COMPOSE_DIR_WSL}' && docker compose up -d"

echo "[evermemos-up] waiting for services (up to ${READINESS_TIMEOUT}s)..."
deadline=$((SECONDS + READINESS_TIMEOUT))
ready=0
while [ $SECONDS -lt $deadline ]; do
  status=$(wsl.exe -d "${WSL_DISTRO}" -- bash -lc "cd '${COMPOSE_DIR_WSL}' && docker compose ps --format '{{.Name}} {{.Health}}'" 2>/dev/null || true)
  unhealthy=$(printf '%s\n' "${status}" | awk 'NF>=2 && $2 != "healthy" && $2 != "" {print}')
  if [ -z "${unhealthy}" ] && [ -n "${status}" ]; then
    ready=1
    break
  fi
  sleep 3
done

if [ $ready -eq 1 ]; then
  echo "[evermemos-up] all services healthy"
else
  echo "[evermemos-up] timeout waiting for healthy services. Current status:" >&2
  wsl.exe -d "${WSL_DISTRO}" -- bash -lc "cd '${COMPOSE_DIR_WSL}' && docker compose ps"
  exit 1
fi
