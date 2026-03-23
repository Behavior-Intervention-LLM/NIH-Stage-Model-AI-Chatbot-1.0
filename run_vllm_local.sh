#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo ".env 不存在，先执行: cp .env.vllm.example .env"
  exit 1
fi

echo "启动 vLLM + FastAPI ..."
docker compose --env-file .env -f docker-compose.vllm.yml up -d --build

echo "服务状态："
docker compose -f docker-compose.vllm.yml ps

echo "健康检查："
echo "- vLLM:  http://127.0.0.1:8001/v1/models"
echo "- API:   http://127.0.0.1:8000/health"
