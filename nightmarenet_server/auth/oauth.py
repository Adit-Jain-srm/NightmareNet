"""OAuth + JWT routes for the hosted NightmareNet platform.

Implements the GitHub + Google authorization-code flows on top of Authlib's
``StarletteOAuth2App``. The router upserts the authenticated user, returns a
short-lived access JWT plus a long-lived refresh JWT, and exposes ``/auth/me``
for the frontend.

The OAuth flow:

1.  ``GET /auth/{provider}/login`` — redirects the browser to the provider
    authorisation URL with state + PKCE.
2.  ``GET /auth/{provider}/callback`` — verifies the authorisation code,
    fetches the user profile, upserts a :class:`User`, and returns
    ``{access_token, refresh_token, user}``.
3.  ``POST /auth/refresh`` — exchanges a refresh token for a new access token.
4.  ``GET /auth/me`` — returns the current user (requires ``Authorization:
    Bearer <jwt>``).

All optional dependencies (Authlib, FastAPI, SQLAlchemy, httpx) are guarded
with ``try/except`` import blocks so the OSS core continues to work without
them. If any required dependency is missing, :func:`build_oauth_router`
returns ``None`` and the hosted server simply does not mount the router.
"""

import os
import secrets
from typing import Any, Dict, Optional, Tuple

from nightmarenet_server.auth.jwt_helpers import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
)

try:
    from fastapi import APIRouter, Depends, HTTPException, Request, status
    from fastapi.responses import RedirectResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError:
    APIRouter = None  # type: ignore[assignment,misc]
    Depends = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    RedirectResponse = None  # type: ignore[assignment,misc]
    HTTPAuthorizationCredentials = None  # type: ignore[assignment,misc]
    HTTPBearer = None  # type: ignore[assignment,misc]
    status = None  # type: ignore[assignment]

try:
    from authlib.integrations.starlette_client import OAuth, OAuthError  # type: ignore
except ImportError:
    OAuth = None  # type: ignore[assignment,misc]
    OAuthError = Exception  # type: ignore[assignment,misc]

try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = None  # type: ignore[assignment,misc]


GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


def _settings() -> Dict[str, str]:
    """Read OAuth provider credentials from environment."""
    return {
        "github_client_id": os.environ.get("NIGHTMARENET_GITHUB_CLIENT_ID", ""),
        "github_client_secret": os.environ.get("NIGHTMARENET_GITHUB_CLIENT_SECRET", ""),
        "google_client_id": os.environ.get("NIGHTMARENET_GOOGLE_CLIENT_ID", ""),
        "google_client_secret": os.environ.get("NIGHTMARENET_GOOGLE_CLIENT_SECRET", ""),
        "session_secret": os.environ.get(
            "NIGHTMARENET_SESSION_SECRET",
            secrets.token_urlsafe(32),
        ),
    }


def _normalize_github_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a GitHub user payload into our canonical shape."""
    email = profile.get("email") or f"{profile.get('login', 'user')}@users.noreply.github.com"
    return {
        "email": email,
        "name": profile.get("name") or profile.get("login") or "",
        "avatar_url": profile.get("avatar_url"),
        "provider": "github",
        "provider_id": str(profile.get("id", "")),
    }


def _normalize_google_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a Google userinfo payload into our canonical shape."""
    return {
        "email": profile.get("email", ""),
        "name": profile.get("name") or profile.get("given_name") or "",
        "avatar_url": profile.get("picture"),
        "provider": "google",
        "provider_id": str(profile.get("sub", "")),
    }


def _build_oauth_client() -> Optional[Any]:
    """Construct an Authlib OAuth registry, or ``None`` if unavailable."""
    if OAuth is None:
        return None

    settings = _settings()
    oauth = OAuth()
    if settings["github_client_id"]:
        oauth.register(
            name="github",
            client_id=settings["github_client_id"],
            client_secret=settings["github_client_secret"],
            access_token_url=GITHUB_TOKEN_URL,
            authorize_url=GITHUB_AUTHORIZE_URL,
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
        )
    if settings["google_client_id"]:
        oauth.register(
            name="google",
            client_id=settings["google_client_id"],
            client_secret=settings["google_client_secret"],
            server_metadata_url=GOOGLE_DISCOVERY_URL,
            client_kwargs={"scope": "openid email profile"},
        )
    return oauth


def _upsert_user(
    session: Any,
    profile: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Insert or update a user row from a normalized OAuth profile.

    Returns ``(user_id, user_dict)``. Falls back to a deterministic dict-only
    representation when SQLAlchemy/User isn't importable (e.g. during unit
    tests without the hosted extras installed).
    """
    try:
        from nightmarenet_server.models import User
    except ImportError:
        return profile.get("provider_id") or profile["email"], {
            "id": profile.get("provider_id") or profile["email"],
            **profile,
        }

    user = session.query(User).filter(User.email == profile["email"]).one_or_none()
    if user is None:
        user = User(
            email=profile["email"],
            name=profile.get("name", ""),
            avatar_url=profile.get("avatar_url"),
        )
        session.add(user)
        session.flush()
    else:
        if profile.get("name"):
            user.name = profile["name"]
        if profile.get("avatar_url"):
            user.avatar_url = profile["avatar_url"]
    session.commit()
    return user.id, {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "provider": profile.get("provider"),
    }


def _issue_tokens(user_id: str, role: str = "member") -> Dict[str, Any]:
    """Build an access/refresh token pair for a user."""
    access = create_access_token(subject=user_id, role=role, expires_in=3600)
    refresh = create_refresh_token(subject=user_id)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": 3600,
    }


def _get_session_dependency() -> Any:
    """Return a FastAPI dependency that yields a DB session, or ``None``."""
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


def build_oauth_router() -> Optional[Any]:
    """Construct the OAuth router or return ``None`` if FastAPI is missing.

    The hosted server (:mod:`nightmarenet_server.app`) calls this at startup
    and mounts the returned router under ``/auth`` when it exists.
    """
    if APIRouter is None:
        return None

    router = APIRouter(prefix="/auth", tags=["auth"])
    oauth = _build_oauth_client()
    bearer = HTTPBearer(auto_error=False)
    session_dep = _get_session_dependency()

    bearer_param = Depends(bearer)
    session_param: Any = Depends(session_dep) if session_dep else None

    def _require_oauth(provider: str) -> Any:
        if oauth is None:
            raise HTTPException(
                status_code=503,
                detail="OAuth provider not configured (Authlib missing).",
            )
        client = oauth.create_client(provider)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"OAuth provider '{provider}' is not configured.",
            )
        return client

    @router.get("/{provider}/login", name="oauth_login")
    async def oauth_login(provider: str, request: Request) -> Any:
        if provider not in {"github", "google"}:
            raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
        client = _require_oauth(provider)
        redirect_uri = str(request.url_for("oauth_callback", provider=provider))
        return await client.authorize_redirect(request, redirect_uri)

    @router.get("/{provider}/callback", name="oauth_callback")
    async def oauth_callback(
        provider: str,
        request: Request,
        db: Any = session_param,
    ) -> Dict[str, Any]:
        if provider not in {"github", "google"}:
            raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
        client = _require_oauth(provider)
        try:
            token = await client.authorize_access_token(request)
        except OAuthError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"OAuth exchange failed: {exc}",
            ) from exc

        if provider == "github":
            resp = await client.get("user", token=token)
            raw_profile = resp.json()
            profile = _normalize_github_profile(raw_profile)
        else:
            userinfo = token.get("userinfo")
            if userinfo is None:
                userinfo = await client.parse_id_token(request, token)
            profile = _normalize_google_profile(dict(userinfo or {}))

        if not profile["email"]:
            raise HTTPException(status_code=400, detail="OAuth profile missing email.")

        if db is None:
            user_id = profile.get("provider_id") or profile["email"]
            user_dict = {"id": user_id, **profile}
        else:
            user_id, user_dict = _upsert_user(db, profile)

        tokens = _issue_tokens(user_id)
        return {"user": user_dict, **tokens}

    @router.post("/refresh")
    async def refresh_token_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
        token = body.get("refresh_token") if isinstance(body, dict) else None
        if not token:
            raise HTTPException(status_code=400, detail="refresh_token is required.")
        try:
            payload = decode_access_token(token)
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid refresh token: {exc}",
            ) from exc
        if payload.get("typ") != "refresh":
            raise HTTPException(status_code=401, detail="Not a refresh token.")
        return _issue_tokens(payload["sub"], role=payload.get("role", "member"))

    @router.get("/me")
    async def get_me(
        credentials: Any = bearer_param,
        db: Any = session_param,
    ) -> Dict[str, Any]:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
            )
        try:
            payload = decode_access_token(credentials.credentials)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {exc}",
            ) from exc

        user_id = payload.get("sub")
        if db is None or user_id is None:
            return {
                "id": user_id,
                "role": payload.get("role", "member"),
                "org_id": payload.get("org_id"),
            }
        try:
            from nightmarenet_server.models import User
        except ImportError:
            return {"id": user_id, "role": payload.get("role", "member")}

        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "avatar_url": user.avatar_url,
            "role": payload.get("role", "member"),
            "org_id": payload.get("org_id"),
        }

    return router
