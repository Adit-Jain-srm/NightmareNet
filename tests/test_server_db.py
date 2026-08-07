import importlib
import os
from unittest import mock

import pytest
from sqlalchemy.engine import Engine

# We will test two modules: nightmarenet_server.db and nightmarenet_server.models.base

@pytest.fixture
def clean_db_env():
    """Ensure environment variables are clean before each test."""
    keys = [
        "NIGHTMARENET_DB_POOL_SIZE",
        "NIGHTMARENET_DB_MAX_OVERFLOW",
        "NIGHTMARENET_DB_POOL_TIMEOUT",
        "NIGHTMARENET_DB_POOL_RECYCLE",
    ]
    original_env = {k: os.environ.get(k) for k in keys}
    
    for k in keys:
        if k in os.environ:
            del os.environ[k]
            
    yield
    
    for k, v in original_env.items():
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v


def test_db_env_defaults(clean_db_env):
    """Test that default values are used when environment is not set."""
    import nightmarenet_server.db as db
    importlib.reload(db)
    
    assert db.DB_POOL_SIZE == 10
    assert db.DB_MAX_OVERFLOW == 20
    assert db.DB_POOL_TIMEOUT == 30
    assert db.DB_POOL_RECYCLE == 3600


def test_db_env_overrides(clean_db_env):
    """Test that valid environment variables override the defaults."""
    os.environ["NIGHTMARENET_DB_POOL_SIZE"] = "15"
    os.environ["NIGHTMARENET_DB_MAX_OVERFLOW"] = "25"
    os.environ["NIGHTMARENET_DB_POOL_TIMEOUT"] = "35"
    os.environ["NIGHTMARENET_DB_POOL_RECYCLE"] = "3605"
    
    import nightmarenet_server.db as db
    importlib.reload(db)
    
    assert db.DB_POOL_SIZE == 15
    assert db.DB_MAX_OVERFLOW == 25
    assert db.DB_POOL_TIMEOUT == 35
    assert db.DB_POOL_RECYCLE == 3605


def test_db_env_invalid_fallback(clean_db_env):
    """Test that invalid environment variables fall back to defaults gracefully."""
    os.environ["NIGHTMARENET_DB_POOL_SIZE"] = "invalid"
    os.environ["NIGHTMARENET_DB_MAX_OVERFLOW"] = "not_an_int"
    
    import nightmarenet_server.db as db
    importlib.reload(db)
    
    assert db.DB_POOL_SIZE == 10
    assert db.DB_MAX_OVERFLOW == 20


def test_engine_sqlite_behavior():
    """Test SQLite engine creation."""
    from nightmarenet_server.models.base import get_engine
    
    engine = get_engine("sqlite:///./test.db")
    assert isinstance(engine, Engine)
    assert engine.url.drivername == "sqlite"
    # SQLite engines shouldn't have pool_size (they use NullPool or SingletonThreadPool depending on args)
    # The important part is that we didn't pass pool_size to create_engine
    assert getattr(engine.pool, "_pool_size", None) is None


@mock.patch("nightmarenet_server.models.base.create_engine")
def test_engine_non_sqlite_arguments(mock_create_engine):
    """Test non-SQLite engine creation passes the right pool arguments."""
    from nightmarenet_server.models.base import get_engine
    import nightmarenet_server.db as db
    
    get_engine("postgresql://user:pass@localhost/db")
    
    mock_create_engine.assert_called_once()
    _, kwargs = mock_create_engine.call_args
    assert kwargs["pool_size"] == db.DB_POOL_SIZE
    assert kwargs["max_overflow"] == db.DB_MAX_OVERFLOW
    assert kwargs["pool_timeout"] == db.DB_POOL_TIMEOUT
    assert kwargs["pool_recycle"] == db.DB_POOL_RECYCLE
    assert kwargs["pool_pre_ping"] is True
