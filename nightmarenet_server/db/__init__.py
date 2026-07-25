"""Database tooling for the hosted NightmareNet platform.

Houses the Alembic migration environment and any future per-database
helpers. The ORM models themselves live in :mod:`nightmarenet_server.models`.
"""

import os

DB_POOL_SIZE = int(os.getenv("NIGHTMARENET_DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("NIGHTMARENET_DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("NIGHTMARENET_DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("NIGHTMARENET_DB_POOL_RECYCLE", "3600"))
