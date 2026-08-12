"""SQLAlchemy declarative base and session helpers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from nightmarenet_server.db import (
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)

DEFAULT_DATABASE_URL = "sqlite:///./nightmarenet_hosted.db"


class Base(DeclarativeBase):
    """Declarative base for hosted platform ORM models."""


def get_engine(database_url: str = DEFAULT_DATABASE_URL):
    """Create a SQLAlchemy engine."""
    connect_args = {}
    kwargs = {}
    if database_url.startswith("sqlite"):
        from sqlalchemy.pool import NullPool
        connect_args["check_same_thread"] = False
        kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            {
                "pool_size": DB_POOL_SIZE,
                "max_overflow": DB_MAX_OVERFLOW,
                "pool_timeout": DB_POOL_TIMEOUT,
                "pool_recycle": DB_POOL_RECYCLE,
                "pool_pre_ping": True,
            }
        )
    return create_engine(database_url, connect_args=connect_args, **kwargs)


def get_session_factory(database_url: str = DEFAULT_DATABASE_URL):
    """Return a configured session factory."""
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """Create all tables (development bootstrap)."""
    from nightmarenet_server.models import tables  # noqa: F401

    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
