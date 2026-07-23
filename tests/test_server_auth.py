from unittest import mock

import pytest

pytest.importorskip("sqlalchemy")

try:
    import jwt
except ImportError:
    jwt = None

from nightmarenet_server.auth import api_keys, jwt_helpers


@pytest.mark.skipif(jwt is None, reason="PyJWT not installed")
def test_jwt_encode_decode():
    token = jwt_helpers.create_access_token(subject="user_123", role="admin")
    assert isinstance(token, str)
    decoded = jwt_helpers.decode_access_token(token)
    assert decoded["sub"] == "user_123"
    assert decoded["role"] == "admin"
    assert decoded["typ"] == "access"

@pytest.mark.skipif(jwt is None, reason="PyJWT not installed")
def test_jwt_expiry():
    with mock.patch("time.time", return_value=1000):
        token = jwt_helpers.create_access_token(subject="user_123", expires_in=10)

    with mock.patch("time.time", return_value=1020):
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, jwt_helpers.get_secret(), algorithms=["HS256"])

@pytest.mark.skipif(jwt is None, reason="PyJWT not installed")
def test_jwt_invalid_token():
    with pytest.raises(jwt.DecodeError):
        jwt_helpers.decode_access_token("invalid.token.string")

def test_generate_api_key():
    plaintext, hashed = api_keys.generate_api_key()
    assert plaintext.startswith("nm_")
    assert len(hashed) == 64  # SHA256 hex digest length
    assert api_keys._hash_key(plaintext) == hashed

def test_mint_api_key():
    mock_session = mock.MagicMock()
    mock_api_key_cls = mock.MagicMock()

    with mock.patch("nightmarenet_server.auth.api_keys.ApiKey", mock_api_key_cls, create=True):
        plaintext, api_key = api_keys.mint_api_key(
            session=mock_session,
            org_id="org_1",
            user_id="user_1",
            name="test_key",
            scopes=["read"]
        )
        assert plaintext.startswith("nm_")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

def test_revoke_api_key():
    mock_session = mock.MagicMock()
    mock_row = mock.MagicMock()
    mock_session.get.return_value = mock_row

    with mock.patch("nightmarenet_server.auth.api_keys.ApiKey", mock.MagicMock(), create=True):
        res = api_keys.revoke_api_key(mock_session, "key_id_123")
        assert res is True
        mock_session.delete.assert_called_once_with(mock_row)
        mock_session.commit.assert_called_once()

def test_revoke_api_key_not_found():
    mock_session = mock.MagicMock()
    mock_session.get.return_value = None

    with mock.patch("nightmarenet_server.auth.api_keys.ApiKey", mock.MagicMock(), create=True):
        res = api_keys.revoke_api_key(mock_session, "key_id_123")
        assert res is False
        mock_session.delete.assert_not_called()
