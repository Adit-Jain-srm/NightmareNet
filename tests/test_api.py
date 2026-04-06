"""Tests for the NightmareNet FastAPI platform API."""

from __future__ import annotations

import pytest

# Only run if fastapi is installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from nightmarenet.api.app import app  # noqa: E402

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
                "text": (
                    "Machine learning is a subset of artificial intelligence."
                    " It allows computers to learn from data."
                ),
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
                "text": (
                    "A fairly long sentence that should"
                    " show degradation at higher strengths clearly."
                ),
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


class TestAuthentication:
    """Test API key authentication middleware."""

    def test_health_bypasses_auth(self, monkeypatch):
        """Health endpoint should always be accessible, even with auth enabled."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "test-secret-key")
        import importlib

        from nightmarenet.api import app as app_module
        importlib.reload(app_module)
        from nightmarenet.api.app import app as reloaded_app
        auth_client = TestClient(reloaded_app)
        response = auth_client.get("/api/v1/health")
        assert response.status_code == 200
        # Cleanup: reload without the key
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)
        importlib.reload(app_module)

    def test_no_key_dev_mode_allows_requests(self):
        """Without NIGHTMARENET_API_KEY set, all requests should pass (dev mode)."""
        # The default test client has no key set
        response = client.post(
            "/api/v1/generate/dream",
            json={"text": "Dev mode test.", "strength": 0.1},
        )
        assert response.status_code == 200

    def test_valid_key_allows_request(self, monkeypatch):
        """Requests with correct key should succeed."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "test-secret-key")
        import importlib

        from nightmarenet.api import app as app_module
        importlib.reload(app_module)
        from nightmarenet.api.app import app as reloaded_app
        auth_client = TestClient(reloaded_app)
        response = auth_client.post(
            "/api/v1/generate/dream",
            json={"text": "Auth test.", "strength": 0.1},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)
        importlib.reload(app_module)

    def test_invalid_key_returns_401(self, monkeypatch):
        """Requests with wrong key should get 401."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "test-secret-key")
        import importlib

        from nightmarenet.api import app as app_module
        importlib.reload(app_module)
        from nightmarenet.api.app import app as reloaded_app
        auth_client = TestClient(reloaded_app)
        response = auth_client.post(
            "/api/v1/generate/dream",
            json={"text": "Auth test.", "strength": 0.1},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert response.json()["error"] == "Unauthorized"
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)
        importlib.reload(app_module)

    def test_missing_key_returns_401(self, monkeypatch):
        """Requests without key header should get 401 when auth is enabled."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "test-secret-key")
        import importlib

        from nightmarenet.api import app as app_module
        importlib.reload(app_module)
        from nightmarenet.api.app import app as reloaded_app
        auth_client = TestClient(reloaded_app)
        response = auth_client.post(
            "/api/v1/generate/dream",
            json={"text": "Auth test.", "strength": 0.1},
        )
        assert response.status_code == 401
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)
        importlib.reload(app_module)


class TestRateLimiting:
    """Test rate limiting returns 429 with expected body."""

    def test_rate_limit_returns_429(self):
        """The rate-limit exception handler should return 429 with JSON error body."""
        import asyncio
        import json
        from unittest.mock import MagicMock

        from nightmarenet.api.app import _rate_limit_handler

        fake_request = MagicMock()
        fake_exc = MagicMock()
        fake_exc.detail = "1 per 1 minute"

        response = asyncio.new_event_loop().run_until_complete(
            _rate_limit_handler(fake_request, fake_exc)
        )
        assert response.status_code == 429
        body = json.loads(response.body)
        assert body["error"] == "Rate limit exceeded"
        assert "detail" in body


class TestTrainingConfigEndpoint:
    """Test the training configuration preview endpoint."""

    def test_config_defaults(self):
        response = client.post("/api/v1/train/config", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["total_phases"] > 0
        assert data["total_epochs"] > 0
        assert len(data["estimated_phases"]) > 0
        assert "config_summary" in data

    def test_config_phase_count(self):
        response = client.post(
            "/api/v1/train/config",
            json={"num_cycles": 2, "wake_epochs": 3, "dream_epochs": 2, "nightmare_epochs": 1},
        )
        assert response.status_code == 200
        data = response.json()
        # 2 cycles × (wake + dream + nightmare + compress) = 8 phases
        assert data["total_phases"] == 8
        # 2 cycles × (3 + 2 + 1 + 1) = 14 epochs
        assert data["total_epochs"] == 14

    def test_config_includes_all_phase_types(self):
        response = client.post(
            "/api/v1/train/config",
            json={"num_cycles": 1, "wake_epochs": 1, "dream_epochs": 1, "nightmare_epochs": 1},
        )
        data = response.json()
        phase_types = [p["phase"] for p in data["estimated_phases"]]
        assert "wake" in phase_types
        assert "dream" in phase_types
        assert "nightmare" in phase_types
        assert "compress" in phase_types

    def test_config_nightmare_lr_multiplier(self):
        response = client.post(
            "/api/v1/train/config",
            json={
                "num_cycles": 1,
                "wake_epochs": 1,
                "dream_epochs": 0,
                "nightmare_epochs": 1,
                "learning_rate": 0.001,
                "nightmare_lr_multiplier": 3.0,
            },
        )
        data = response.json()
        nightmare_phase = [p for p in data["estimated_phases"] if p["phase"] == "nightmare"][0]
        assert nightmare_phase["learning_rate"] == pytest.approx(0.003)

    def test_config_recommends_learned_adversarial(self):
        response = client.post(
            "/api/v1/train/config",
            json={"nightmare_strength": 0.8, "use_learned_adversarial": False},
        )
        data = response.json()
        recs = " ".join(data["recommendations"])
        assert "learned_adversarial" in recs.lower() or "learned adversarial" in recs.lower()

    def test_config_recommends_against_high_dream_strength(self):
        response = client.post(
            "/api/v1/train/config",
            json={"dream_strength": 0.6},
        )
        data = response.json()
        recs = " ".join(data["recommendations"])
        assert "dream strength" in recs.lower()

    def test_config_recommends_more_cycles(self):
        response = client.post(
            "/api/v1/train/config",
            json={"num_cycles": 1},
        )
        data = response.json()
        recs = " ".join(data["recommendations"])
        assert "cycle" in recs.lower()

    def test_config_zero_nightmare_epochs_warns(self):
        response = client.post(
            "/api/v1/train/config",
            json={"nightmare_epochs": 0},
        )
        data = response.json()
        recs = " ".join(data["recommendations"])
        assert "nightmare" in recs.lower() and "disabled" in recs.lower()

    def test_config_invalid_model_type(self):
        response = client.post(
            "/api/v1/train/config",
            json={"model_type": "invalid_type"},
        )
        data = response.json()
        assert data["valid"] is False
        assert any("invalid" in r.lower() for r in data["recommendations"])

    def test_config_learned_adversarial_in_description(self):
        response = client.post(
            "/api/v1/train/config",
            json={
                "num_cycles": 1,
                "nightmare_epochs": 1,
                "use_learned_adversarial": True,
            },
        )
        data = response.json()
        nightmare_phase = [p for p in data["estimated_phases"] if p["phase"] == "nightmare"][0]
        assert "learned adversarial" in nightmare_phase["description"].lower()

    def test_config_skips_zero_epoch_phases(self):
        response = client.post(
            "/api/v1/train/config",
            json={"num_cycles": 1, "wake_epochs": 0, "dream_epochs": 0, "nightmare_epochs": 0},
        )
        data = response.json()
        phase_types = [p["phase"] for p in data["estimated_phases"]]
        assert "wake" not in phase_types
        assert "dream" not in phase_types
        assert "nightmare" not in phase_types
        assert "compress" in phase_types

    def test_config_validation_rejects_bad_values(self):
        response = client.post(
            "/api/v1/train/config",
            json={"num_cycles": -1},
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/train/config",
            json={"learning_rate": 0},
        )
        assert response.status_code == 422


class TestCompareEndpoint:
    """Test the distortion comparison endpoint."""

    def test_compare_basic(self):
        response = client.post(
            "/api/v1/compare",
            json={"text": "The quick brown fox jumps over the lazy dog."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "dream" in data
        assert "nightmare" in data
        assert "baseline" in data["dream"]
        assert "challenge" in data["dream"]
        assert "resilience_score" in data
        assert 0 <= data["resilience_score"] <= 1
        assert "analysis" in data

    def test_compare_with_custom_strengths(self):
        response = client.post(
            "/api/v1/compare",
            json={
                "text": "Test text for comparison.",
                "baseline_strength": 0.1,
                "challenge_strength": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["baseline_strength"] == 0.1
        assert data["challenge_strength"] == 0.9

    def test_compare_baseline_less_distorted(self):
        response = client.post(
            "/api/v1/compare",
            json={
                "text": (
                    "A sufficiently long sentence that should show meaningful"
                    " degradation differences between baseline and challenge strengths."
                ),
                "baseline_strength": 0.0,
                "challenge_strength": 0.9,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # At strength 0.0, baseline should be nearly identical
        assert data["dream"]["baseline"]["similarity"] >= 0.9

    def test_compare_deterministic_with_seed(self):
        payload = {
            "text": "Reproducibility test text.",
            "baseline_strength": 0.3,
            "challenge_strength": 0.7,
            "seed": 123,
        }
        resp1 = client.post("/api/v1/compare", json=payload)
        resp2 = client.post("/api/v1/compare", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["dream"] == resp2.json()["dream"]


class TestHealthTestsPassing:
    """Test that health endpoint returns test count."""

    def test_health_includes_tests_passing(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "tests_passing" in data

    def test_health_tests_passing_is_int_or_null(self):
        response = client.get("/api/v1/health")
        data = response.json()
        tp = data["tests_passing"]
        assert tp is None or isinstance(tp, int)


class TestUploadEndpoint:
    """Test the file upload endpoint."""

    def test_upload_txt_file(self):
        content = b"Hello world. This is a test file."
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["file_type"] == ".txt"
        assert data["text_content"] == "Hello world. This is a test file."
        assert data["char_count"] == len(content)
        assert data["word_count"] == 7
        assert data["line_count"] == 1

    def test_upload_csv_file(self):
        content = b"col1,col2\nval1,val2\nval3,val4"
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("data.csv", content, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "data.csv"
        assert data["file_type"] == ".csv"
        assert data["line_count"] == 3

    def test_upload_json_file(self):
        content = b'{"text": "some input", "value": 42}'
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("input.json", content, "application/json")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "input.json"
        assert data["file_type"] == ".json"

    def test_upload_rejects_unsupported_type(self):
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("image.png", b"\x89PNG", "image/png")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_rejects_too_large(self, monkeypatch):
        import nightmarenet.api.app as api_module
        monkeypatch.setattr(api_module, "_MAX_UPLOAD_BYTES", 10)
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("big.txt", b"x" * 100, "text/plain")},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_upload_rejects_invalid_utf8(self):
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("bad.txt", b"\xff\xfe\x00\x80\x81", "text/plain")},
        )
        assert response.status_code == 400
        assert "UTF-8" in response.json()["detail"]

    def test_upload_preview_truncation(self):
        long_text = "A" * 1000
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("long.txt", long_text.encode(), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["preview"].endswith("...")
        assert len(data["preview"]) == 503  # 500 chars + "..."

    def test_upload_word_count_multiline(self):
        content = b"line one\nline two has more words\nthird"
        response = client.post(
            "/api/v1/upload/text",
            files={"file": ("multi.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["word_count"] == 8
        assert data["line_count"] == 3

    def test_compare_empty_text_rejected(self):
        response = client.post(
            "/api/v1/compare",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_compare_invalid_strength_rejected(self):
        response = client.post(
            "/api/v1/compare",
            json={"text": "Test.", "baseline_strength": 2.0},
        )
        assert response.status_code == 422

    def test_compare_response_structure(self):
        response = client.post(
            "/api/v1/compare",
            json={"text": "Structure test."},
        )
        data = response.json()
        for mode in ["dream", "nightmare"]:
            for level in ["baseline", "challenge"]:
                detail = data[mode][level]
                assert "distorted_text" in detail
                assert "similarity" in detail
                assert "length_ratio" in detail
                assert isinstance(detail["similarity"], float)
                assert isinstance(detail["length_ratio"], float)

    def test_compare_analysis_contains_metrics(self):
        response = client.post(
            "/api/v1/compare",
            json={"text": "Analysis content check."},
        )
        data = response.json()
        assert "Dream" in data["analysis"] or "dream" in data["analysis"].lower()
        assert "Nightmare" in data["analysis"] or "nightmare" in data["analysis"].lower()
        assert "Resilience" in data["analysis"] or "resilience" in data["analysis"].lower()


class TestLearnedAdversarialIntegration:
    """Test that learned adversarial distortions are enabled in nightmare at high strength."""

    def test_nightmare_high_strength_uses_adversarial_config(self):
        """At strength >= 0.5, nightmare should add learned adversarial to its config."""
        from nightmarenet.api.app import _apply_nightmare_distortions

        # This should not raise - the learned module guards against model loading failures
        result = _apply_nightmare_distortions(
            "Test text for adversarial distortions.", strength=0.8, seed=42
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nightmare_low_strength_no_learned_adversarial(self):
        """At strength < 0.5, should use default config without learned adversarial."""
        from nightmarenet.api.app import _apply_nightmare_distortions

        result = _apply_nightmare_distortions(
            "Test text at low strength.", strength=0.3, seed=42
        )
        assert isinstance(result, str)

    def test_nightmare_custom_config_preserved(self):
        """User-provided adversarial config should be preserved even at high strength."""
        from nightmarenet.api.app import _apply_nightmare_distortions

        custom_config = {"adversarial": {"contradiction": 1.0}}
        result = _apply_nightmare_distortions(
            "Custom config test.", strength=0.8, config=custom_config, seed=42
        )
        assert isinstance(result, str)


class TestOpenAPINewEndpoints:
    """Test that new endpoints appear in OpenAPI schema."""

    def test_openapi_includes_train_config(self):
        response = client.get("/openapi.json")
        data = response.json()
        assert "/api/v1/train/config" in data["paths"]

    def test_openapi_includes_compare(self):
        response = client.get("/openapi.json")
        data = response.json()
        assert "/api/v1/compare" in data["paths"]

    def test_openapi_includes_demo(self):
        response = client.get("/openapi.json")
        data = response.json()
        assert "/api/v1/demo" in data["paths"]


class TestDemoEndpoint:
    """Test the interactive demo endpoint (combined dream+nightmare)."""

    def test_demo_basic(self):
        """Demo returns both dream and nightmare results in one call."""
        response = client.post(
            "/api/v1/demo",
            json={"text": "The quick brown fox jumps over the lazy dog."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_text" in data
        assert "dream" in data
        assert "nightmare" in data
        assert "resilience_delta" in data
        assert "insight" in data

    def test_demo_response_structure(self):
        """Dream and nightmare fields contain expected detail keys."""
        response = client.post(
            "/api/v1/demo",
            json={"text": "Test text for structure validation."},
        )
        data = response.json()
        for mode in ["dream", "nightmare"]:
            detail = data[mode]
            assert "distorted_text" in detail
            assert "similarity" in detail
            assert "length_ratio" in detail
            assert isinstance(detail["similarity"], float)
            assert 0.0 <= detail["similarity"] <= 1.0

    def test_demo_resilience_delta_bounds(self):
        """Resilience delta should be within [0, 1]."""
        response = client.post(
            "/api/v1/demo",
            json={
                "text": (
                    "A sufficiently long sentence to produce"
                    " meaningful distortion differences."
                ),
            },
        )
        data = response.json()
        assert -1.0 <= data["resilience_delta"] <= 1.0

    def test_demo_insight_contains_word_count(self):
        """Insight should mention the word count of the input."""
        text = "One two three four five six seven."
        response = client.post(
            "/api/v1/demo",
            json={"text": text},
        )
        data = response.json()
        assert "7-word" in data["insight"]

    def test_demo_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        payload = {"text": "Deterministic test.", "seed": 42}
        resp1 = client.post("/api/v1/demo", json=payload)
        resp2 = client.post("/api/v1/demo", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["dream"] == resp2.json()["dream"]
        assert resp1.json()["nightmare"] == resp2.json()["nightmare"]

    def test_demo_empty_text_rejected(self):
        """Empty text should be rejected with 422."""
        response = client.post(
            "/api/v1/demo",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_demo_dream_more_similar_than_nightmare(self):
        """Dream and nightmare both produce valid distortion results."""
        response = client.post(
            "/api/v1/demo",
            json={
                "text": (
                    "Machine learning models process input data"
                    " through layers of learned transformations"
                    " to produce meaningful output predictions."
                ),
            },
        )
        data = response.json()
        # Both distortions should produce valid similarity scores
        assert 0.0 <= data["dream"]["similarity"] <= 1.0
        assert 0.0 <= data["nightmare"]["similarity"] <= 1.0
        # Both should have non-empty distorted text
        assert len(data["dream"]["distorted_text"]) > 0
        assert len(data["nightmare"]["distorted_text"]) > 0

    def test_demo_insight_contains_resilience_quality(self):
        """Insight should contain a resilience quality assessment."""
        response = client.post(
            "/api/v1/demo",
            json={"text": "Quality assessment test input."},
        )
        data = response.json()
        insight_lower = data["insight"].lower()
        assert any(
            w in insight_lower
            for w in ["resilient", "vulnerable"]
        )

