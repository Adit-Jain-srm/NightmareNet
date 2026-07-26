"""JWT helpers for hosted platform authentication (scaffold)."""

import os
import time
from typing import Any, Dict, Optional

# Optional dependency — install sqlalchemy + pyjwt for hosted server
try:
    import jwt
except ImportError:
    jwt = None  # type: ignore


def get_secret() -> str:
    """Return JWT signing secret from environment."""
    secret = os.environ.get("NIGHTMARENET_JWT_SECRET")
    if not secret:
        raise RuntimeError("NIGHTMARENET_JWT_SECRET environment variable is not set")
    return secret


def create_access_token(
    subject: str,
    org_id: Optional[str] = None,
    role: str = "member",
    expires_in: int = 3600,
    token_type: str = "access",
) -> str:
    """Create a signed JWT access token."""
    if jwt is None:
        raise RuntimeError("PyJWT is required for token creation. pip install PyJWT")

    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_in,
        "role": role,
        "typ": token_type,
    }
    if org_id:
        payload["org_id"] = org_id
    return jwt.encode(payload, get_secret(), algorithm="HS256")


def create_refresh_token(
    subject: str,
    expires_in: int = 60 * 60 * 24 * 30,
) -> str:
    """Create a long-lived refresh token (default 30 days)."""
    return create_access_token(
        subject=subject,
        expires_in=expires_in,
        token_type="refresh",
    )


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    if jwt is None:
        raise RuntimeError("PyJWT is required. pip install PyJWT")
    return jwt.decode(token, get_secret(), algorithms=["HS256"])
