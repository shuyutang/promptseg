#!/usr/bin/env bash
# Build:  docker build -t sam2web-backend backend/
# Run:    ./backend/start_server.sh   then open http://localhost:8000
set -euo pipefail

mkdir -p "${HOME}/.cache/sam2web-weights"

docker run --rm -it --gpus all \
  -p 8000:8000 \
  -v "${HOME}/.cache/sam2web-weights:/weights" \
  -e SAM2_MODEL_ID="${SAM2_MODEL_ID:-facebook/sam2.1-hiera-base-plus}" \
  --name sam2api \
  sam2web-backend
