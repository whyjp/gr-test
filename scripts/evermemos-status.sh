#!/usr/bin/env bash
# Show EverMemOS container status + basic health endpoints.

set -euo pipefail

WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR_HOST="${REPO_ROOT}/external/everos/methods/evermemos"
COMPOSE_DIR_WSL=$(echo "${COMPOSE_DIR_HOST}" | sed -E 's|^([A-Za-z]):/|/mnt/\L\1/|')

echo "=== docker compose ps ==="
wsl.exe -d "${WSL_DISTRO}" -- bash -lc "cd '${COMPOSE_DIR_WSL}' && docker compose ps"

echo ""
echo "=== Port checks (from host) ==="
for name_port in "mongo:27017" "es:19200" "milvus:19530" "redis:6379" "api:1995"; do
  name="${name_port%:*}"
  port="${name_port##*:}"
  if (echo > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    printf "  %-8s (%5s) OK\n" "${name}" "${port}"
  else
    printf "  %-8s (%5s) --\n" "${name}" "${port}"
  fi
done

echo ""
echo "=== EverMemOS API /health ==="
if command -v curl >/dev/null 2>&1; then
  curl -sS -m 3 http://localhost:1995/health || echo "(no response)"
else
  echo "curl not available"
fi
