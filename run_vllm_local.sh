#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo ".env not found. Run: cp .env.vllm.example .env"
  exit 1
fi

echo "Starting vLLM + FastAPI ..."
docker compose --env-file .env -f docker-compose.vllm.yml up -d --build

echo "Service status:"
docker compose -f docker-compose.vllm.yml ps

echo "Health checks:"
echo "- vLLM:  http://127.0.0.1:8001/v1/models"
echo "- API:   http://127.0.0.1:8000/health"
