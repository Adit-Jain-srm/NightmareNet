"""API key minting, hashing, and Bearer-token verification.

Keys are presented to clients as ``nm_<base62-token>`` and stored in the
``api_keys`` table as SHA-256 digests of the raw token. The plaintext is
only returned at mint time — subsequent lookups always go through the hash.

Public surface:

* :func:`generate_api_key` — create a fresh ``(plaintext, hash)`` pair.
* :func:`mint_api_key` — persist a new key for a user and return the secret.
* :func:`revoke_api_key` — soft-delete a key by id.
* :func:`require_api_key` — FastAPI dependency that resolves the bearer
  token to its ``ApiKey`` row (or 401s).

FastAPI / SQLAlchemy are imported with the project's standard ``try/except``
fallback so this module is safely importable without the hosted extras.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

try:
    from fastapi import Depends, HTTPException, Request, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError:
    Depends = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    status = None  # type: ignore[assignment]
    HTTPAuthorizationCredentials = None  # type: ignore[assignment,misc]
    HTTPBearer = None  # type: ignore[assignment,misc]

try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = None  # type: ignore[assignment,misc]


API_KEY_PREFIX = "nm_"
_TOKEN_BYTES = 32


def _hash_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest of an API key plaintext."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_api_key() -> Tuple[str, str]:
    """Generate a fresh ``(plaintext, hash)`` API-key pair.

    The plaintext is the value handed to the user; only the hash is persisted.
    """
    token = secrets.token_urlsafe(_TOKEN_BYTES).rstrip("=")
    plaintext = f"{API_KEY_PREFIX}{token}"
    return plaintext, _hash_key(plaintext)


def mint_api_key(
    session: Any,
    org_id: str,
    user_id: str,
    name: str = "default",
    scopes: Optional[List[str]] = None,
) -> Tuple[str, Any]:
    """Persist a new API key and return ``(plaintext, ApiKey)``.

    The plaintext is only returned here — callers must surface it to the
    end-user immediately because it cannot be recovered later.
    """
    try:
        from nightmarenet_server.models import ApiKey
    except ImportError as exc:
        raise RuntimeError("nightmarenet_server.models is required to mint API keys") from exc

    plaintext, hashed = generate_api_key()
    api_key = ApiKey(
        org_id=org_id,
        user_id=user_id,
        key_hash=hashed,
        name=name,
        scopes=json.dumps(scopes or []),
    )
    session.add(api_key)
    session.commit()
    session.refresh(api_key)
    return plaintext, api_key


def revoke_api_key(session: Any, api_key_id: str) -> bool:
    """Delete an API key by id. Returns True if a row was removed."""
    try:
        from nightmarenet_server.models import ApiKey
    except ImportError as exc:
        raise RuntimeError("nightmarenet_server.models is required to revoke API keys") from exc

    row = session.get(ApiKey, api_key_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


def lookup_api_key(session: Any, plaintext: str) -> Optional[Any]:
    """Return the :class:`ApiKey` row matching ``plaintext``, or ``None``."""
    try:
        from nightmarenet_server.models import ApiKey
    except ImportError:
        return None

    if not plaintext or not plaintext.startswith(API_KEY_PREFIX):
        return None

    hashed = _hash_key(plaintext)
    api_key = session.query(ApiKey).filter(ApiKey.key_hash == hashed).one_or_none()
    if api_key is None:
        return None
    api_key.last_used_at = datetime.now(timezone.utc)
    session.commit()
    return api_key


def _get_session_dependency() -> Any:
    """Return a generator dependency that yields a hosted DB session."""
    try:
        from nightmarenet_server.models.base import (
            DEFAULT_DATABASE_URL,
            get_session_factory,
        )
    except ImportError:
        return None

    db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
    session_factory = get_session_factory(db_url)

    def _dep() -> Any:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    return _dep


def _extract_bearer(request: Any) -> Optional[str]:
    """Pull a Bearer token from an ``Authorization`` header, if present."""
    if request is None:
        return None
    header = request.headers.get("authorization") if hasattr(request, "headers") else None
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def require_api_key() -> Any:
    """Build a FastAPI dependency that resolves the bearer token to an ApiKey.

    Returns ``None`` when FastAPI isn't available; this lets the OSS package
    import the module without pulling in the hosted dependencies.
    """
    if Depends is None or HTTPException is None:
        return None

    bearer = HTTPBearer(auto_error=False)
    session_dep = _get_session_dependency()

    bearer_param = Depends(bearer)
    session_param: Any = Depends(session_dep) if session_dep else None

    async def _dependency(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = bearer_param,
        db: Any = session_param,
    ) -> Any:
        token = credentials.credentials if credentials else _extract_bearer(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not token.startswith(API_KEY_PREFIX):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key must start with '{API_KEY_PREFIX}'.",
            )
        if db is None:
            raise HTTPException(
                status_code=503,
                detail="API key store is not configured.",
            )
        api_key = lookup_api_key(db, token)
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key.",
            )
        return api_key

    return _dependency
