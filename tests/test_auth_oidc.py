"""Unit tests for enterprise OIDC SSO (issue #700)."""

from __future__ import annotations

import time
from unittest import mock

import pytest

pytest.importorskip("jwt")

import jwt

from nightmarenet_server.auth import oidc


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("NIGHTMARENET_JWT_SECRET", "test-secret-oidc-32bytes-minimum!!")


def test_generate_pkce_pair_s256():
    verifier, challenge = oidc.generate_pkce_pair()
    assert len(verifier) > 20
    assert challenge != verifier
    # Recompute challenge to confirm S256
    import base64
    import hashlib

    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_discover_provider_parses_metadata():
    metadata = {
        "issuer": "https://login.example.com/tenant",
        "authorization_endpoint": "https://login.example.com/authorize",
        "token_endpoint": "https://login.example.com/token",
        "jwks_uri": "https://login.example.com/jwks",
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return metadata

    class _Client:
        def get(self, url):
            assert "openid-configuration" in url
            return _Resp()

    parsed = oidc.discover_provider("https://login.example.com/tenant", client=_Client())
    assert parsed["issuer"] == metadata["issuer"]
    assert parsed["jwks_uri"] == metadata["jwks_uri"]


def test_validate_id_token_accepts_valid_and_rejects_bad_audience():
    now = int(time.time())
    key = "unit-test-hs256-key-32bytes-min!!"
    good = jwt.encode(
        {
            "sub": "user-1",
            "email": "a@example.com",
            "iss": "https://issuer.example",
            "aud": "client-123",
            "iat": now,
            "exp": now + 600,
        },
        key,
        algorithm="HS256",
    )
    claims = oidc.validate_id_token(
        good,
        audience="client-123",
        issuer="https://issuer.example",
        key=key,
    )
    assert claims["sub"] == "user-1"

    with pytest.raises(jwt.InvalidAudienceError):
        oidc.validate_id_token(
            good,
            audience="other-client",
            issuer="https://issuer.example",
            key=key,
        )


def test_validate_id_token_rejects_expired():
    key = "unit-test-hs256-key-32bytes-min!!"
    token = jwt.encode(
        {
            "sub": "user-1",
            "iss": "https://issuer.example",
            "aud": "client-123",
            "iat": 1_000,
            "exp": 1_010,
        },
        key,
        algorithm="HS256",
    )
    with mock.patch("time.time", return_value=2_000):
        with pytest.raises(jwt.ExpiredSignatureError):
            oidc.validate_id_token(
                token,
                audience="client-123",
                issuer="https://issuer.example",
                key=key,
                leeway=0,
            )


def test_map_groups_to_role_and_claims():
    assert oidc.map_groups_to_role(["NightmareNet-Admins"]) == "admin"
    assert oidc.map_groups_to_role(["unknown"]) == "member"
    profile = oidc.map_oidc_claims(
        {
            "sub": "oid-99",
            "email": "admin@corp.com",
            "name": "Ada",
            "groups": ["NightmareNet-Admins"],
        },
        provider="azure",
    )
    assert profile["provider"] == "azure"
    assert profile["provider_id"] == "oid-99"
    assert profile["role"] == "admin"
    assert profile["email"] == "admin@corp.com"


def test_upsert_sso_user_jit_provisioning():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.orm import sessionmaker

    from nightmarenet_server.models.base import Base, get_engine
    from nightmarenet_server.models.tables import User

    engine = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    profile = {
        "email": "first@corp.com",
        "name": "First",
        "provider": "okta",
        "provider_id": "ext-1",
        "role": "member",
    }
    user_id, user = oidc.upsert_sso_user(session, profile)
    assert user_id
    assert user["email"] == "first@corp.com"
    row = session.query(User).filter(User.id == user_id).one()
    assert row.sso_provider == "okta"
    assert row.external_id == "ext-1"

    # Second login updates name, same identity
    profile["name"] = "First Updated"
    user_id2, user2 = oidc.upsert_sso_user(session, profile)
    assert user_id2 == user_id
    assert user2["name"] == "First Updated"
    session.close()


def test_issue_sso_tokens_default_eight_hours(monkeypatch):
    monkeypatch.setenv("NIGHTMARENET_SSO_SESSION_SECONDS", "28800")
    tokens = oidc.issue_sso_tokens("user-x", role="admin", org_id="org-1")
    assert tokens["expires_in"] == 28800
    assert tokens["token_type"] == "bearer"
    decoded = jwt.decode(
        tokens["access_token"],
        "test-secret-oidc-32bytes-minimum!!",
        algorithms=["HS256"],
    )
    assert decoded["sub"] == "user-x"
    assert decoded["role"] == "admin"
    assert decoded["org_id"] == "org-1"


def test_build_authorize_url_includes_pkce():
    metadata = {"authorization_endpoint": "https://idp.example/authorize"}
    url = oidc.build_authorize_url(
        metadata,
        client_id="cid",
        redirect_uri="https://app.example/callback",
        state="st",
        code_challenge="chal",
    )
    assert "code_challenge=chal" in url
    assert "code_challenge_method=S256" in url
    assert "client_id=cid" in url


def test_load_sso_provider_env_fallback(monkeypatch):
    monkeypatch.setenv(
        "NIGHTMARENET_OIDC_DEFAULT_METADATA_URL",
        "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
    )
    monkeypatch.setenv("NIGHTMARENET_OIDC_CLIENT_ID", "azure-client")
    cfg = oidc.load_sso_provider(None)
    assert cfg["client_id"] == "azure-client"
    assert "openid-configuration" in cfg["metadata_url"]


def test_sso_login_returns_503_when_unconfigured(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware

    monkeypatch.delenv("NIGHTMARENET_OIDC_CLIENT_ID", raising=False)
    monkeypatch.delenv("NIGHTMARENET_OIDC_DEFAULT_METADATA_URL", raising=False)
    monkeypatch.setenv("NIGHTMARENET_DATABASE_URL", "sqlite:///:memory:")

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test")
    router = oidc.build_sso_router()
    assert router is not None
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/v1/auth/sso/login")
    assert resp.status_code == 503
