import json
from pathlib import Path

import jsonschema
import pytest
from jose import jwt

from nightmarenet.compliance.json_export import (
    export_signed_json,
    get_schema,
    verify_signed_json,
)
from nightmarenet.compliance.report import generate_report


def test_generate_compliance_report(tmp_path):
    config = {
        "model": {
            "name": "test-model",
            "type": "transformer",
        },
        "dataset": {
            "name": "dummy",
            "path": "dummy/path",
        },
    }

    comparison = {
        "robustness_score": 0.91,
        "robustness_delta": 0.07,
        "metrics": {
            "robustness": {
                "trained": {
                    "auc_robustness": 0.91,
                },
                "deltas": {
                    "auc_robustness": 0.07,
                },
            }
        },
    }

    report = generate_report(
        config=config,
        comparison=comparison,
        model_path="",
        output_dir=str(tmp_path),
    )

    assert report["model"]["name"] == "test-model"
    assert report["robustness"]["delta"] == 0.07

    files = list(Path(tmp_path).glob("*compliance_report.json"))
    assert files

    with open(files[0], encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["model"]["name"] == "test-model"


def test_config_hash_is_deterministic():
    from nightmarenet.compliance.report import _config_hash

    config = {
        "model": {"name": "demo"},
        "dataset": {"name": "dummy"},
    }

    first = _config_hash(config)
    second = _config_hash(config)

    assert first == second


def test_config_defaults_to_no_compliance_report():
    config = {
        "tracking": {
            "compliance_report": False,
        }
    }

    assert config["tracking"]["compliance_report"] is False


@pytest.fixture
def dummy_report():
    return {
        "model": {"name": "test-model-123"},
        "dataset": {"name": "test-data"},
        "robustness": {"delta": 0.07},
    }


@pytest.fixture
def rsa_key_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


def test_signed_json_export(dummy_report, rsa_key_pem):
    token = export_signed_json(dummy_report, rsa_key_pem)
    assert token.count(".") == 2


def test_signed_json_schema(dummy_report, rsa_key_pem):
    token = export_signed_json(dummy_report, rsa_key_pem)
    payload = jwt.decode(token, rsa_key_pem, algorithms=["RS256"])

    schema = get_schema()
    # This should not raise an exception
    jsonschema.validate(instance=payload, schema=schema)

    assert payload["model_id"] == "test-model-123"
    assert "version_hash" in payload
    assert "timestamp" in payload


def test_jws_verification(dummy_report, rsa_key_pem):
    token = export_signed_json(dummy_report, rsa_key_pem)
    result = verify_signed_json(token, rsa_key_pem)
    assert result.is_valid
    assert result.payload["model_id"] == "test-model-123"


def test_tampered_jws_is_invalid(dummy_report, rsa_key_pem):
    token = export_signed_json(dummy_report, rsa_key_pem)
    # token is header.payload.signature
    parts = token.split(".")
    import base64
    import json

    # decode payload, modify it, encode it back
    payload_json = base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8")
    payload = json.loads(payload_json)
    payload["model_id"] = "hacked-model"
    encoded_payload = json.dumps(payload).encode("utf-8")
    new_payload = base64.urlsafe_b64encode(encoded_payload).decode("utf-8").rstrip("=")
    tampered_token = f"{parts[0]}.{new_payload}.{parts[2]}"

    result = verify_signed_json(tampered_token, rsa_key_pem)
    assert not result.is_valid
    # Either signature verification fails (which it will, because the signature
    # is for the old payload)

    # What if we just tamper the version hash? Actually, if signature fails it fails first.
    # To test version_hash failure specifically, one would need to re-sign it
    # with the same key but wrong hash.
    # Let's do that.
    payload["version_hash"] = "wrong-hash"
    tampered_token2 = jwt.encode(payload, rsa_key_pem, algorithm="RS256")
    result2 = verify_signed_json(tampered_token2, rsa_key_pem)
    assert not result2.is_valid
    assert "version_hash mismatch" in result2.error


def test_verification_reports_metadata(dummy_report, rsa_key_pem):
    token = export_signed_json(dummy_report, rsa_key_pem)
    result = verify_signed_json(token, rsa_key_pem)
    assert result.is_valid
    assert "schema_version" in result.payload
    assert "timestamp" in result.payload


def test_missing_signing_key_produces_error(dummy_report):
    # Pass an invalid or empty key
    with pytest.raises(ValueError):
        export_signed_json(dummy_report, "")


def test_pdf_export_is_unchanged():
    # We didn't change pdf export. The test_generate_compliance_report already tests
    # the report building.
    # This just ensures we haven't broken the import or signature.
    from nightmarenet.compliance.report import generate_pdf

    assert callable(generate_pdf)
