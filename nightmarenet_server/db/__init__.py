"""Database tooling for the hosted NightmareNet platform.

Houses the Alembic migration environment and any future per-database
helpers. The ORM models themselves live in :mod:`nightmarenet_server.models`.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _parse_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid value for %s: %s. Using default %d.", key, val, default)
        return default


DB_POOL_SIZE = _parse_int_env("NIGHTMARENET_DB_POOL_SIZE", 10)
DB_MAX_OVERFLOW = _parse_int_env("NIGHTMARENET_DB_MAX_OVERFLOW", 20)
DB_POOL_TIMEOUT = _parse_int_env("NIGHTMARENET_DB_POOL_TIMEOUT", 30)
DB_POOL_RECYCLE = _parse_int_env("NIGHTMARENET_DB_POOL_RECYCLE", 3600)
