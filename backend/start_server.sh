#!/usr/bin/env bash
# Build:  docker build -t promptseg-backend backend/
# Run:    ./backend/start_server.sh   then open http://localhost:8000
set -euo pipefail

mkdir -p "${HOME}/.cache/promptseg-weights"

# Sessions live on the host, not in the container: --rm would otherwise throw
# away every annotation the moment the server stops, which is the one thing
# saving them is meant to prevent.
DATA_DIR="${SAM2_DATA_DIR:-${HOME}/.local/share/promptseg}"
mkdir -p "${DATA_DIR}"

docker run --rm -it --gpus all \
  -p 8000:8000 \
  -v "${HOME}/.cache/promptseg-weights:/weights" \
  -v "${DATA_DIR}:/data" \
  -e SAM2_MODEL_ID="${SAM2_MODEL_ID:-facebook/sam2.1-hiera-base-plus}" \
  -e SAM2_DATA_DIR=/data \
  -e SAM2_PERSIST="${SAM2_PERSIST:-1}" \
  --name sam2api \
  promptseg-backend
