#!/usr/bin/env bash
set -euo pipefail

APP_NAME="appoint-ready-gpu"
CONTEXT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${CONTEXT_DIR}/env.list"

echo "[info] Build context: ${CONTEXT_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] Docker not found. Please install Docker." >&2
  exit 1
fi

echo "[info] Building GPU image..."
docker build -f "${CONTEXT_DIR}/Dockerfile.gpu" -t "${APP_NAME}" "${CONTEXT_DIR}"

echo "[info] Running container with --gpus all"
RUN_ARGS=(
  --rm -it \
  --gpus all \
  -p 7860:7860 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
)

if [ -f "${ENV_FILE}" ]; then
  RUN_ARGS+=( --env-file "${ENV_FILE}" )
else
  echo "[warn] ${ENV_FILE} not found. Using defaults baked into the image."
fi

docker run "${RUN_ARGS[@]}" "${APP_NAME}"

