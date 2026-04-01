"""Tests for the NightmareNet FastAPI platform API."""

from __future__ import annotations

import pytest

# Only run if fastapi is installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from nightmarenet.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_includes_version(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "0.2.0"


class TestDreamEndpoint:
    """Test the dream distortion generation endpoint."""

    def test_dream_basic(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "The quick brown fox jumps over the lazy dog.", "strength": 0.3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["distortion_type"] == "dream"
        assert data["original_text"] == "The quick brown fox jumps over the lazy dog."
        assert isinstance(data["distorted_text"], str)
        assert data["strength"] == 0.3

    def test_dream_with_seed(self):
        resp1 = client.post(
            "/api/v1/generate/dream",
            json={"text": "Hello world.", "strength": 0.5, "seed": 42},
        )
        resp2 = client.post(
            "/api/v1/generate/dream",
            json={"text": "Hello world.", "strength": 0.5, "seed": 42},
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["distorted_text"] == resp2.json()["distorted_text"]

    def test_dream_zero_strength(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "Unchanged text.", "strength": 0.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["distorted_text"] == "Unchanged text."

    def test_dream_empty_text_rejected(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "", "strength": 0.3},
        )
        assert response.status_code == 422  # Pydantic validation

    def test_dream_invalid_strength_rejected(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "Test.", "strength": 1.5},
        )
        assert response.status_code == 422

    def test_dream_negative_strength_rejected(self):
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "Test.", "strength": -0.1},
        )
        assert response.status_code == 422


class TestNightmareEndpoint:
    """Test the nightmare distortion generation endpoint."""

    def test_nightmare_basic(self):
        response = client.post(
            "/api/v1/generate/nightmare",
            json={
                "text": "Machine learning is a subset of artificial intelligence. It allows computers to learn from data.",
                "strength": 0.8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["distortion_type"] == "nightmare"
        assert isinstance(data["distorted_text"], str)

    def test_nightmare_high_strength(self):
        response = client.post(
            "/api/v1/generate/nightmare",
            json={"text": "The weather is nice today.", "strength": 0.9},
        )
        assert response.status_code == 200

    def test_nightmare_with_config(self):
        response = client.post(
            "/api/v1/generate/nightmare",
            json={
                "text": "Test with custom config.",
                "strength": 0.5,
                "config": {"char_swap": 1.0},
            },
        )
        assert response.status_code == 200


class TestRobustnessEndpoint:
    """Test the robustness evaluation endpoint."""

    def test_robustness_basic(self):
        response = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": "The quick brown fox jumps over the lazy dog."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert "dream" in data["scores"]
        assert "nightmare" in data["scores"]
        assert "summary" in data

    def test_robustness_custom_strengths(self):
        response = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": "Test text.", "strengths": [0.1, 0.5, 0.9]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]["dream"]) == 3

    def test_robustness_similarity_decreases_with_strength(self):
        response = client.post(
            "/api/v1/evaluate/robustness",
            json={
                "text": "A fairly long sentence that should show degradation at higher strengths clearly.",
                "strengths": [0.0, 0.5, 1.0],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # At strength 0.0, similarity should be very high
        low_sim = data["scores"]["dream"]["0.0"]["similarity"]
        assert low_sim >= 0.9  # Nearly identical at zero strength

    def test_robustness_empty_text_rejected(self):
        response = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": ""},
        )
        assert response.status_code == 422


class TestOpenAPIDocs:
    """Test that API documentation is accessible."""

    def test_docs_endpoint(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "NightmareNet API"
        assert "/api/v1/generate/dream" in data["paths"]
        assert "/api/v1/generate/nightmare" in data["paths"]
        assert "/api/v1/evaluate/robustness" in data["paths"]
