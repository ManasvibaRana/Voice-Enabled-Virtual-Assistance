"""Make `app` importable and guarantee the database exists before tests run."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as app_config  # noqa: E402


def pytest_configure(config):  # noqa: ARG001 - name fixed by the pytest hookspec
    if not app_config.DB_PATH.exists():
        print("[tests] seeding data/app.db ...")
        subprocess.run(
            [sys.executable, str(ROOT / "data" / "seed_db.py")],
            check=True,
        )
