"""SQLAlchemy declarative base and session helpers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./nightmarenet_hosted.db"


class Base(DeclarativeBase):
    """Declarative base for hosted platform ORM models."""


def get_engine(database_url: str = DEFAULT_DATABASE_URL):
    """Create a SQLAlchemy engine."""
    connect_args = {}
    kwargs = {}
    if database_url.startswith("sqlite"):
        from sqlalchemy.pool import QueuePool

        connect_args["check_same_thread"] = False
        kwargs["poolclass"] = QueuePool
        kwargs["pool_size"] = 1
        kwargs["max_overflow"] = 0
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
