@echo off
REM Seed the database (if needed), then start the server.
REM Document ingestion happens automatically on server startup.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

if not exist "data\app.db" (
    echo [run] seeding data\app.db ...
    python data\seed_db.py || goto :error
)

if not exist ".env" (
    echo [run] WARNING: .env not found. Copy .env.example to .env and paste your
    echo [run]          free NVIDIA NIM key from https://build.nvidia.com
)

echo [run] starting server on http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo [run] seeding failed.
exit /b 1
