#!/bin/bash
set -e

source venv/bin/activate 2>/dev/null || true

echo "Starting backend on http://localhost:8000 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:8501 ..."
streamlit run frontend_streamlit.py &
FRONTEND_PID=$!

trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
wait
