#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo ".env not found. Run: cp .env.local.example .env"
  exit 1
fi

echo "Starting local development service (FastAPI)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
