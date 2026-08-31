#!/usr/bin/env python3
"""
check_migrations.py
-------------------
Validates that all Alembic migrations apply cleanly (upgrade head) and can be
fully reversed (downgrade base) against a fresh, isolated SQLite database.

Usage
-----
    python scripts/check_migrations.py

Environment
-----------
``NIGHTMARENET_DATABASE_URL`` is temporarily set to an isolated SQLite file
(``ci_migration_test.db``) so the script never touches any real database.
The temporary database file is removed on exit (success *and* failure).

Exit codes
----------
0 — Both directions completed without errors.
1 — At least one direction failed; details printed to stderr.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Isolated DB file — created fresh, deleted on exit.
_DB_PATH = Path("ci_migration_test.db")
_DB_URL = f"sqlite:///{_DB_PATH}"


def _run(command: list[str], description: str) -> None:
    """Run *command* with the test DB URL injected; exit 1 on failure."""
    print(f"\n→ {description}", flush=True)
    result = subprocess.run(
        command,
        env={**os.environ, "NIGHTMARENET_DATABASE_URL": _DB_URL},
    )
    if result.returncode != 0:
        print(f"\n✗ FAILED: {description}", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"✓ OK: {description}", flush=True)


def _cleanup() -> None:
    """Remove the temporary SQLite database file if it exists."""
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError as exc:
        # Non-fatal — warn only.
        print(f"Warning: could not remove {_DB_PATH}: {exc}", file=sys.stderr)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Always start from a clean slate.
    _cleanup()

    try:
        _run(["alembic", "upgrade", "head"], "alembic upgrade head")
        _run(["alembic", "downgrade", "base"], "alembic downgrade base")
        print("\n✅ All migrations validated successfully.", flush=True)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
