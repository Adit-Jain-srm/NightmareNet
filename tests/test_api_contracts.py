"""API contract tests: validate live responses against the committed OpenAPI spec.

These tests complement the OpenAPI *drift* check in CI
(``scripts/export_openapi.py --check``, which verifies the committed
``docs/api/openapi.json`` matches the generated schema) by verifying that the
FastAPI app's *runtime responses* actually conform to the response schemas
documented in that spec. Schema drift between code and documentation causes
client-side parsing failures that would otherwise only surface in production.

The spec is loaded from ``docs/api/openapi.json`` at test time, and responses
are validated with the ``jsonschema`` library. All tests run against a
``TestClient`` with no external services (no DB, Redis, or network).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Only run when the API extra (and jsonschema) are available.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
jsonschema = pytest.importorskip("jsonschema")

from fastapi.testclient import TestClient  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402

from nightmarenet.api.app import app  # noqa: E402

client = TestClient(app)

SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "api" / "openapi.json"

# Base URI the spec is registered under so fragment-only refs
# ("#/components/schemas/...") resolve during validation.
SPEC_URI = "urn:nightmarenet-openapi"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    """The committed OpenAPI spec (source of truth for contract validation)."""
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry(spec: dict[str, Any]) -> Registry:
    """A referencing registry so ``$ref`` component schemas resolve during validation."""
    resource = Resource.from_contents(spec, default_specification=DRAFT202012)
    return Registry().with_resource(uri=SPEC_URI, resource=resource)


def _response_schema_ref(spec: dict[str, Any], path: str, method: str, status: str) -> str:
    """Return the component ``$ref`` documented for a given path/method/status code."""
    responses = spec["paths"][path][method]["responses"]
    assert status in responses, f"{method.upper()} {path} has no documented {status} response"
    content = responses[status].get("content", {})
    assert "application/json" in content, (
        f"{method.upper()} {path} {status} does not document an application/json body"
    )
    schema = content["application/json"]["schema"]
    ref = schema.get("$ref")
    assert ref and ref.startswith("#/"), (
        f"{method.upper()} {path} {status} response schema is not a component $ref: {schema!r}"
    )
    return ref


def _validator(ref: str, registry: Registry) -> Draft202012Validator:
    """Build a validator for a fragment ``$ref`` rebased onto the registered spec."""
    return Draft202012Validator({"$ref": SPEC_URI + ref}, registry=registry)


def _assert_valid_ref(instance: Any, ref: str, registry: Registry) -> None:
    """Validate ``instance`` against a fragment ``$ref`` from the spec."""
    validator = _validator(ref, registry)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    assert not errors, "Schema validation errors:\n" + "\n".join(
        f"  - {list(e.absolute_path)}: {e.message}" for e in errors
    )


class TestSuccessResponseContracts:
    """2xx responses must conform to their documented schemas."""

    def test_health_matches_schema(self, spec, registry):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/health", "get", "200")
        _assert_valid_ref(response.json(), ref, registry)

    def test_dream_matches_schema(self, spec, registry):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "The quick brown fox.", "strength": 0.3},
        )
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/generate/dream", "post", "200")
        _assert_valid_ref(response.json(), ref, registry)

    def test_nightmare_matches_schema(self, spec, registry):
        response = client.post(
            "/api/v1/generate/nightmare",
            json={"text": "Machine learning is a subset of AI.", "strength": 0.8},
        )
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/generate/nightmare", "post", "200")
        _assert_valid_ref(response.json(), ref, registry)

    def test_robustness_matches_schema(self, spec, registry):
        response = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": "The quick brown fox jumps over the lazy dog."},
        )
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/evaluate/robustness", "post", "200")
        _assert_valid_ref(response.json(), ref, registry)

    def test_train_config_matches_schema(self, spec, registry):
        response = client.post("/api/v1/train/config", json={})
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/train/config", "post", "200")
        _assert_valid_ref(response.json(), ref, registry)


class TestRequiredFieldsPresent:
    """Required fields declared in the spec must never be null/missing in 2xx bodies."""

    def test_health_required_fields(self, spec):
        data = client.get("/api/v1/health").json()
        required = spec["components"]["schemas"]["HealthResponse"].get("required", [])
        assert "version" in required
        for field in required:
            assert field in data and data[field] is not None, f"missing required field {field!r}"

    def test_dream_required_fields(self, spec):
        data = client.post(
            "/api/v1/generate/dream",
            json={"text": "Contract test.", "strength": 0.3},
        ).json()
        required = spec["components"]["schemas"]["DistortionResponse"].get("required", [])
        assert required, "DistortionResponse should declare required fields"
        for field in required:
            assert field in data and data[field] is not None, f"missing required field {field!r}"


class TestContentTypeContracts:
    """Response Content-Type headers must match what the spec documents."""

    def test_success_content_type_is_json(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    def test_validation_error_content_type_is_json(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "", "strength": 0.3},
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")


class TestErrorResponseContracts:
    """4xx responses must conform to their documented / actual error schemas."""

    def test_validation_error_matches_http_validation_error(self, spec, registry):
        # Empty text violates the min_length constraint -> documented 422.
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "", "strength": 0.3},
        )
        assert response.status_code == 422
        ref = _response_schema_ref(spec, "/api/v1/generate/dream", "post", "422")
        assert ref.endswith("HTTPValidationError")
        _assert_valid_ref(response.json(), ref, registry)

    def test_invalid_body_returns_422(self):
        # strength out of range is rejected by request-body validation.
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "Test.", "strength": 1.5},
        )
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body and isinstance(body["detail"], list)

    def test_not_found_error_shape(self):
        # A missing resource returns FastAPI's standard {"detail": <str>} body.
        response = client.get("/api/v1/compliance/report/does_not_exist_run")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert isinstance(body, dict)
        assert isinstance(body.get("detail"), str)


class TestPaginationContract:
    """List endpoints must return the documented pagination shape."""

    def test_pipeline_runs_pagination_shape(self, spec, registry):
        response = client.get("/api/v1/pipeline/runs")
        assert response.status_code == 200
        ref = _response_schema_ref(spec, "/api/v1/pipeline/runs", "get", "200")
        _assert_valid_ref(response.json(), ref, registry)

        data = response.json()
        for field in ("runs", "total", "offset", "limit"):
            assert field in data, f"pagination field {field!r} missing"
        assert isinstance(data["runs"], list)


class TestSpecCompleteness:
    """A new endpoint added without updating the committed spec must fail here."""

    @staticmethod
    def _spec_routes(spec: dict[str, Any]) -> set[tuple[str, str]]:
        routes: set[tuple[str, str]] = set()
        for path, methods in spec["paths"].items():
            for method in methods:
                if method.lower() in {"get", "post", "put", "patch", "delete"}:
                    routes.add((path, method.lower()))
        return routes

    @staticmethod
    def _app_routes() -> set[tuple[str, str]]:
        routes: set[tuple[str, str]] = set()
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods or not path.startswith("/api/"):
                continue
            for method in methods:
                if method.lower() in {"get", "post", "put", "patch", "delete"}:
                    routes.add((path, method.lower()))
        return routes

    def test_every_runtime_api_route_is_in_spec(self, spec):
        missing = self._app_routes() - self._spec_routes(spec)
        assert not missing, (
            "These runtime API routes are missing from docs/api/openapi.json "
            "(regenerate with `make openapi`): " + ", ".join(sorted(str(m) for m in missing))
        )

    def test_spec_matches_runtime_openapi_paths(self):
        # The committed spec must document exactly the app's advertised paths.
        with open(SPEC_PATH, encoding="utf-8") as f:
            committed = json.load(f)
        live = app.openapi()
        assert set(committed["paths"]) == set(live["paths"]), (
            "Committed spec paths differ from the live app; run `make openapi`."
        )
