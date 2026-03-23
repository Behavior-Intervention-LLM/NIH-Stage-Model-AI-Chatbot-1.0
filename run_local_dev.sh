#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo ".env 不存在，先执行: cp .env.local.example .env"
  exit 1
fi

echo "启动本地开发服务（FastAPI）..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
