#!/usr/bin/env bash
# Seed the database (if needed), then start the server.
# Document ingestion happens automatically on server startup.
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
fi

if [ ! -f "data/app.db" ]; then
  echo "[run] seeding data/app.db ..."
  python data/seed_db.py
fi

if [ ! -f ".env" ]; then
  echo "[run] WARNING: .env not found. Copy .env.example to .env and paste your"
  echo "[run]          free NVIDIA NIM key from https://build.nvidia.com"
fi

echo "[run] starting server on http://127.0.0.1:8000"
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
