"""Enterprise SSO via OpenID Connect (Azure AD / Okta / Auth0).

Provides OIDC discovery, PKCE, ID-token validation, JIT user provisioning,
group→role mapping, and FastAPI routes under ``/api/v1/auth/sso``.

Coordinates with :mod:`nightmarenet_server.auth.oauth` (social login) without
duplicating the GitHub/Google Authlib flows. Application sessions are still
issued as HS256 JWTs via :mod:`nightmarenet_server.auth.jwt_helpers`; IdP
tokens are validated separately (issuer, audience, expiry, signature).
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from nightmarenet_server.auth.jwt_helpers import create_access_token, create_refresh_token

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request
    from fastapi.responses import RedirectResponse
except ImportError:
    APIRouter = None  # type: ignore[assignment,misc]
    Depends = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment,misc]
    Query = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    RedirectResponse = None  # type: ignore[assignment,misc]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import jwt
    from jwt import PyJWKClient
except ImportError:
    jwt = None  # type: ignore[assignment]
    PyJWKClient = None  # type: ignore[assignment,misc]

DEFAULT_SSO_SESSION_SECONDS = 8 * 60 * 60  # 8 hours
DEFAULT_ROLE_MAPPING = {
    "NightmareNet-Admins": "admin",
    "NightmareNet-Members": "member",
    "admins": "admin",
}


def sso_session_seconds() -> int:
    """Configurable session lifetime (default 8h)."""
    raw = os.environ.get("NIGHTMARENET_SSO_SESSION_SECONDS", str(DEFAULT_SSO_SESSION_SECONDS))
    try:
        return max(300, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_SSO_SESSION_SECONDS


def generate_pkce_pair() -> Tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def metadata_url_for_issuer(issuer: str) -> str:
    """Build the standard OIDC discovery URL for an issuer."""
    base = issuer.rstrip("/")
    if base.endswith("/.well-known/openid-configuration"):
        return base
    return f"{base}/.well-known/openid-configuration"


def discover_provider(issuer_or_metadata_url: str, *, client: Any = None) -> Dict[str, Any]:
    """Fetch and return OIDC provider metadata (authorization/token/jwks/issuer)."""
    if httpx is None and client is None:
        raise RuntimeError("httpx is required for OIDC discovery")

    url = issuer_or_metadata_url.strip()
    if not url.endswith("openid-configuration"):
        url = metadata_url_for_issuer(url)

    if client is not None:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
    else:
        with httpx.Client(timeout=15.0) as http:
            response = http.get(url)
            response.raise_for_status()
            data = response.json()

    required = ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"OIDC metadata missing keys: {missing}")
    return data


def build_authorize_url(
    metadata: Dict[str, Any],
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    scope: str = "openid profile email",
    extra: Optional[Dict[str, str]] = None,
) -> str:
    """Construct the OIDC authorization redirect URL (PKCE S256)."""
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if extra:
        params.update(extra)
    return f"{metadata['authorization_endpoint']}?{urlencode(params)}"


def exchange_code(
    metadata: Dict[str, Any],
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    client_secret: Optional[str] = None,
    http_client: Any = None,
) -> Dict[str, Any]:
    """Exchange an authorization code for tokens (supports public PKCE clients)."""
    if httpx is None and http_client is None:
        raise RuntimeError("httpx is required for OIDC token exchange")

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    auth = None
    if client_secret:
        # Confidential clients may also send client_secret
        data["client_secret"] = client_secret

    if http_client is not None:
        response = http_client.post(metadata["token_endpoint"], data=data, auth=auth)
        response.raise_for_status()
        return response.json()

    with httpx.Client(timeout=15.0) as http:
        response = http.post(metadata["token_endpoint"], data=data)
        response.raise_for_status()
        return response.json()


def validate_id_token(
    id_token: str,
    *,
    audience: str,
    issuer: str,
    jwks_uri: Optional[str] = None,
    key: Any = None,
    algorithms: Optional[List[str]] = None,
    leeway: int = 60,
) -> Dict[str, Any]:
    """Validate an OIDC ID token (signature, exp, aud, iss).

    Pass ``key`` for HS256 unit tests; otherwise fetch signing keys from
    ``jwks_uri`` via PyJWKClient.
    """
    if jwt is None:
        raise RuntimeError("PyJWT is required for OIDC ID token validation")

    options = {"require": ["exp", "iat", "iss", "aud", "sub"]}
    algos = algorithms or (["HS256"] if key is not None else ["RS256"])

    if key is not None:
        return jwt.decode(
            id_token,
            key=key,
            algorithms=algos,
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options=options,
        )

    if not jwks_uri:
        raise ValueError("jwks_uri or key is required to validate ID tokens")
    if PyJWKClient is None:
        raise RuntimeError("PyJWT[crypto] / PyJWKClient required for JWKS validation")

    jwks_client = PyJWKClient(jwks_uri)
    signing_key = jwks_client.get_signing_key_from_jwt(id_token)
    return jwt.decode(
        id_token,
        signing_key.key,
        algorithms=algos,
        audience=audience,
        issuer=issuer,
        leeway=leeway,
        options=options,
    )


def map_groups_to_role(
    groups: List[str],
    mapping: Optional[Dict[str, str]] = None,
    *,
    default_role: str = "member",
) -> str:
    """Map IdP group claims to NightmareNet roles (admin/member)."""
    role_map = mapping or DEFAULT_ROLE_MAPPING
    for group in groups:
        if group in role_map:
            return role_map[group]
    return default_role


def map_oidc_claims(
    claims: Dict[str, Any],
    *,
    provider: str,
    role_claim: str = "groups",
    role_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Normalize ID-token claims into a JIT provisioning profile."""
    groups_raw = claims.get(role_claim) or claims.get("groups") or []
    if isinstance(groups_raw, str):
        groups = [groups_raw]
    elif isinstance(groups_raw, list):
        groups = [str(g) for g in groups_raw]
    else:
        groups = []

    role = map_groups_to_role(groups, role_mapping)
    email = claims.get("email") or claims.get("preferred_username") or ""
    if not email and claims.get("sub"):
        email = f"{claims['sub']}@{provider}.sso.local"

    return {
        "email": email,
        "name": claims.get("name") or claims.get("given_name") or email.split("@")[0],
        "avatar_url": claims.get("picture"),
        "provider": provider,
        "provider_id": str(claims.get("sub", "")),
        "role": role,
        "groups": groups,
    }


def upsert_sso_user(session: Any, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """JIT-provision a user by (sso_provider, external_id), falling back to email."""
    try:
        from nightmarenet_server.models import User
    except ImportError:
        uid = profile.get("provider_id") or profile["email"]
        return uid, {"id": uid, **profile}

    provider = profile.get("provider")
    external_id = profile.get("provider_id")
    user = None
    if provider and external_id:
        user = (
            session.query(User)
            .filter(User.sso_provider == provider, User.external_id == external_id)
            .one_or_none()
        )
    if user is None and profile.get("email"):
        user = session.query(User).filter(User.email == profile["email"]).one_or_none()

    if user is None:
        user = User(
            email=profile["email"],
            name=profile.get("name", ""),
            avatar_url=profile.get("avatar_url"),
            sso_provider=provider,
            external_id=external_id,
        )
        session.add(user)
        session.flush()
    else:
        if profile.get("name"):
            user.name = profile["name"]
        if profile.get("avatar_url"):
            user.avatar_url = profile["avatar_url"]
        user.sso_provider = provider
        user.external_id = external_id

    session.commit()
    return user.id, {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "provider": provider,
        "external_id": external_id,
        "role": profile.get("role", "member"),
    }


def issue_sso_tokens(
    user_id: str,
    *,
    role: str = "member",
    org_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Issue app session JWTs bound to the configured SSO lifetime."""
    expires_in = sso_session_seconds()
    access = create_access_token(
        subject=user_id,
        role=role,
        org_id=org_id,
        expires_in=expires_in,
    )
    refresh = create_refresh_token(subject=user_id, expires_in=max(expires_in * 4, expires_in))
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


def _env_provider_config() -> Optional[Dict[str, Any]]:
    """Optional single-tenant SSO config from environment (local/dev)."""
    metadata = os.environ.get("NIGHTMARENET_OIDC_DEFAULT_METADATA_URL", "").strip()
    client_id = os.environ.get("NIGHTMARENET_OIDC_CLIENT_ID", "").strip()
    if not metadata or not client_id:
        return None
    return {
        "id": "env-default",
        "org_id": os.environ.get("NIGHTMARENET_OIDC_DEFAULT_ORG_ID") or None,
        "name": "env-default",
        "issuer": metadata,
        "metadata_url": metadata,
        "client_id": client_id,
        "client_secret": os.environ.get("NIGHTMARENET_OIDC_CLIENT_SECRET") or None,
        "role_claim": os.environ.get("NIGHTMARENET_OIDC_ROLE_CLAIM", "groups"),
        "role_mapping_json": os.environ.get("NIGHTMARENET_OIDC_ROLE_MAPPING", "{}"),
        "enabled": True,
    }


def _provider_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "id": row.id,
        "org_id": row.org_id,
        "name": row.name,
        "issuer": row.issuer,
        "metadata_url": row.metadata_url,
        "client_id": row.client_id,
        "client_secret": row.client_secret,
        "role_claim": row.role_claim,
        "role_mapping_json": row.role_mapping_json,
        "enabled": bool(row.enabled),
    }


def load_sso_provider(
    session: Any,
    *,
    org_id: Optional[str] = None,
    provider_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load an enabled SSO provider for an org, or fall back to env config."""
    if session is not None:
        try:
            from nightmarenet_server.models import SsoProvider
        except ImportError:
            pass
        else:
            try:
                query = session.query(SsoProvider).filter(SsoProvider.enabled.is_(True))
                if provider_id:
                    query = query.filter(SsoProvider.id == provider_id)
                if org_id:
                    query = query.filter(SsoProvider.org_id == org_id)
                row = query.order_by(SsoProvider.created_at.asc()).first()
                if row is not None:
                    return _provider_to_dict(row)
            except Exception:
                logger.debug("SSO provider DB lookup failed; trying env fallback", exc_info=True)

    env_cfg = _env_provider_config()
    if env_cfg is not None:
        return env_cfg
    raise LookupError("No SSO provider configured for this organization")


def _parse_role_mapping(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if not raw:
        return dict(DEFAULT_ROLE_MAPPING)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data:
            return {str(k): str(v) for k, v in data.items()}
    except (TypeError, ValueError):
        pass
    return dict(DEFAULT_ROLE_MAPPING)


def _get_session_dependency() -> Any:
    try:
        from nightmarenet_server.models.base import DEFAULT_DATABASE_URL, get_session_factory
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


def build_sso_router() -> Optional[Any]:
    """Construct enterprise SSO routes, or ``None`` if FastAPI is missing."""
    if APIRouter is None:
        return None

    router = APIRouter(prefix="/api/v1/auth/sso", tags=["sso"])
    session_dep = _get_session_dependency()
    session_param: Any = Depends(session_dep) if session_dep and Depends else None

    @router.get("/login")
    async def sso_login(
        request: Request,
        org_id: Optional[str] = Query(None),
        provider_id: Optional[str] = Query(None),
        db: Any = session_param,
    ) -> Any:
        try:
            provider = load_sso_provider(db, org_id=org_id, provider_id=provider_id)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        meta_url = provider.get("metadata_url") or provider["issuer"]
        try:
            metadata = discover_provider(meta_url)
        except Exception as exc:
            logger.exception("OIDC discovery failed")
            raise HTTPException(status_code=502, detail=f"OIDC discovery failed: {exc}") from exc

        verifier, challenge = generate_pkce_pair()
        state = secrets.token_urlsafe(24)
        redirect_uri = str(request.url_for("sso_callback"))

        # Stash PKCE + provider context in the signed session cookie
        request.session["oidc_state"] = state
        request.session["oidc_verifier"] = verifier
        request.session["oidc_provider"] = {
            "id": provider.get("id"),
            "org_id": provider.get("org_id"),
            "client_id": provider["client_id"],
            "client_secret": provider.get("client_secret"),
            "issuer": metadata["issuer"],
            "jwks_uri": metadata["jwks_uri"],
            "token_endpoint": metadata["token_endpoint"],
            "authorization_endpoint": metadata["authorization_endpoint"],
            "role_claim": provider.get("role_claim", "groups"),
            "role_mapping_json": provider.get("role_mapping_json", "{}"),
            "name": provider.get("name", "oidc"),
        }
        request.session["oidc_redirect_uri"] = redirect_uri

        url = build_authorize_url(
            metadata,
            client_id=provider["client_id"],
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=challenge,
        )
        return RedirectResponse(url)

    @router.get("/callback", name="sso_callback")
    async def sso_callback(
        request: Request,
        code: Optional[str] = None,
        state: Optional[str] = None,
        db: Any = session_param,
    ) -> Dict[str, Any]:
        if not code or not state:
            raise HTTPException(status_code=400, detail="Missing code or state")
        expected_state = request.session.get("oidc_state")
        verifier = request.session.get("oidc_verifier")
        provider_ctx = request.session.get("oidc_provider")
        redirect_uri = request.session.get("oidc_redirect_uri")
        if not expected_state or state != expected_state or not verifier or not provider_ctx:
            raise HTTPException(status_code=400, detail="Invalid or expired SSO state")

        metadata = {
            "issuer": provider_ctx["issuer"],
            "jwks_uri": provider_ctx["jwks_uri"],
            "token_endpoint": provider_ctx["token_endpoint"],
            "authorization_endpoint": provider_ctx["authorization_endpoint"],
        }
        try:
            token_payload = exchange_code(
                metadata,
                client_id=provider_ctx["client_id"],
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=verifier,
                client_secret=provider_ctx.get("client_secret"),
            )
        except Exception as exc:
            logger.exception("OIDC token exchange failed")
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}") from exc

        id_token = token_payload.get("id_token")
        if not id_token:
            raise HTTPException(status_code=400, detail="Provider response missing id_token")

        try:
            claims = validate_id_token(
                id_token,
                audience=provider_ctx["client_id"],
                issuer=provider_ctx["issuer"],
                jwks_uri=provider_ctx["jwks_uri"],
            )
        except Exception as exc:
            logger.exception("OIDC ID token validation failed")
            raise HTTPException(status_code=401, detail=f"Invalid ID token: {exc}") from exc

        profile = map_oidc_claims(
            claims,
            provider=provider_ctx.get("name") or "oidc",
            role_claim=provider_ctx.get("role_claim", "groups"),
            role_mapping=_parse_role_mapping(provider_ctx.get("role_mapping_json")),
        )
        user_id, user = upsert_sso_user(db, profile)
        tokens = issue_sso_tokens(
            user_id,
            role=profile.get("role", "member"),
            org_id=provider_ctx.get("org_id"),
        )

        # Clear one-time PKCE state
        for key in ("oidc_state", "oidc_verifier", "oidc_provider", "oidc_redirect_uri"):
            request.session.pop(key, None)

        return {"user": user, **tokens}

    return router


def build_sso_admin_router() -> Optional[Any]:
    """Admin API to configure SSO providers per organization."""
    if APIRouter is None:
        return None

    admin = APIRouter(prefix="/api/v1/orgs", tags=["sso-admin"])
    session_dep = _get_session_dependency()
    session_param: Any = Depends(session_dep) if session_dep and Depends else None

    @admin.get("/{org_id}/sso-providers")
    async def list_providers(org_id: str, db: Any = session_param) -> Dict[str, Any]:
        from nightmarenet_server.models import SsoProvider

        rows = db.query(SsoProvider).filter(SsoProvider.org_id == org_id).all()
        items = []
        for row in rows:
            item = _provider_to_dict(row)
            item.pop("client_secret", None)
            items.append(item)
        return {"items": items, "total": len(items)}

    @admin.post("/{org_id}/sso-providers")
    async def create_provider(
        org_id: str,
        body: Dict[str, Any],
        db: Any = session_param,
    ) -> Dict[str, Any]:
        import uuid

        from nightmarenet_server.models import SsoProvider

        issuer = body.get("issuer") or body.get("metadata_url")
        client_id = body.get("client_id")
        if not issuer or not client_id:
            raise HTTPException(
                status_code=400,
                detail="issuer/metadata_url and client_id required",
            )
        row = SsoProvider(
            id=str(uuid.uuid4()),
            org_id=org_id,
            name=body.get("name", "default"),
            issuer=issuer,
            metadata_url=body.get("metadata_url") or metadata_url_for_issuer(issuer),
            client_id=client_id,
            client_secret=body.get("client_secret"),
            role_claim=body.get("role_claim", "groups"),
            role_mapping_json=json.dumps(body.get("role_mapping") or DEFAULT_ROLE_MAPPING),
            enabled=bool(body.get("enabled", True)),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        out = _provider_to_dict(row)
        out.pop("client_secret", None)
        return out

    @admin.delete("/{org_id}/sso-providers/{provider_id}")
    async def delete_provider(
        org_id: str,
        provider_id: str,
        db: Any = session_param,
    ) -> Dict[str, Any]:
        from nightmarenet_server.models import SsoProvider

        row = (
            db.query(SsoProvider)
            .filter(SsoProvider.org_id == org_id, SsoProvider.id == provider_id)
            .one_or_none()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="SSO provider not found")
        db.delete(row)
        db.commit()
        return {"id": provider_id, "deleted": True}

    return admin


def build_sso_routers() -> Tuple[Optional[Any], Optional[Any]]:
    """Return ``(sso_router, admin_router)`` for mounting on the hosted app."""
    return build_sso_router(), build_sso_admin_router()
