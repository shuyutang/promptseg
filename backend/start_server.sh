#!/usr/bin/env bash
# Build:  docker build -t promptseg-backend backend/
# Run:    ./backend/start_server.sh   then open http://localhost:8000
set -euo pipefail

mkdir -p "${HOME}/.cache/promptseg-weights"

docker run --rm -it --gpus all \
  -p 8000:8000 \
  -v "${HOME}/.cache/promptseg-weights:/weights" \
  -e SAM2_MODEL_ID="${SAM2_MODEL_ID:-facebook/sam2.1-hiera-base-plus}" \
  --name sam2api \
  promptseg-backend
