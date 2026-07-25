"""Extended edge-case tests for NightmareNet.

Covers empty/null inputs, unicode (CJK + emoji), very long inputs,
and API schema validation errors.
"""

from __future__ import annotations

import random

import pytest

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions

# Only run API tests if fastapi is installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from nightmarenet.api.app import app  # noqa: E402

client = TestClient(app)

SEED = 42

# Very long text (5000+ characters)
VERY_LONG_TEXT = " ".join(["NightmareNet distortion testing sentence number"] * 250)

# Unicode including CJK, emoji, and accented characters
UNICODE_CJK = "これは日本語のテストです。機械学習は素晴らしい技術です。"
UNICODE_EMOJI = "Testing 🎉🚀🔥💀🌊 with emojis and special chars: ñ ü ö ä ß"
UNICODE_MIXED = "Hello 世界! 🌐 Résumé naïve Zürich Ñoño 日本語 한국어"


class TestEmptyStringDistortions:
    """Ensure all distortion functions handle empty string gracefully."""

    def test_text_distortion_empty_returns_empty(self):
        result = apply_text_distortions("", strength=0.5)
        assert result == ""

    def test_semantic_distortion_empty_returns_empty(self):
        result = apply_semantic_distortions("", strength=0.5)
        assert result == ""

    def test_adversarial_distortion_empty_returns_empty(self):
        result = apply_adversarial_distortions("", strength=0.5)
        assert result == ""

    def test_text_distortion_empty_high_strength(self):
        result = apply_text_distortions("", strength=1.0)
        assert result == ""

    def test_semantic_distortion_empty_high_strength(self):
        result = apply_semantic_distortions("", strength=1.0)
        assert result == ""

    def test_adversarial_distortion_empty_high_strength(self):
        result = apply_adversarial_distortions("", strength=1.0)
        assert result == ""


class TestUnicodeThroughPipeline:
    """Unicode (emoji, CJK) passes through all distortion stages."""

    def test_text_distortion_cjk(self):
        random.seed(SEED)
        result = apply_text_distortions(UNICODE_CJK, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_semantic_distortion_cjk(self):
        random.seed(SEED)
        result = apply_semantic_distortions(UNICODE_CJK, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adversarial_distortion_cjk(self):
        random.seed(SEED)
        result = apply_adversarial_distortions(UNICODE_CJK, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_distortion_emoji(self):
        random.seed(SEED)
        result = apply_text_distortions(UNICODE_EMOJI, strength=0.4)
        assert isinstance(result, str)

    def test_semantic_distortion_emoji(self):
        random.seed(SEED)
        result = apply_semantic_distortions(UNICODE_EMOJI, strength=0.4)
        assert isinstance(result, str)

    def test_adversarial_distortion_emoji(self):
        random.seed(SEED)
        result = apply_adversarial_distortions(UNICODE_EMOJI, strength=0.4)
        assert isinstance(result, str)

    def test_text_distortion_mixed_unicode(self):
        random.seed(SEED)
        result = apply_text_distortions(UNICODE_MIXED, strength=0.5)
        assert isinstance(result, str)

    def test_semantic_distortion_mixed_unicode(self):
        random.seed(SEED)
        result = apply_semantic_distortions(UNICODE_MIXED, strength=0.5)
        assert isinstance(result, str)

    def test_adversarial_distortion_mixed_unicode(self):
        random.seed(SEED)
        result = apply_adversarial_distortions(UNICODE_MIXED, strength=0.5)
        assert isinstance(result, str)


class TestVeryLongInput:
    """Inputs with 5000+ characters are handled without crash."""

    def test_long_text_char_count(self):
        assert len(VERY_LONG_TEXT) > 5000

    def test_text_distortion_long_input(self):
        result = apply_text_distortions(VERY_LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_semantic_distortion_long_input(self):
        result = apply_semantic_distortions(VERY_LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adversarial_distortion_long_input(self):
        result = apply_adversarial_distortions(VERY_LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0


class TestAPISchemaValidation:
    """Null/None/missing inputs to API schemas return proper 422 errors."""

    def test_dream_missing_text_field(self):
        resp = client.post("/api/v1/generate/dream", json={"strength": 0.5})
        assert resp.status_code == 422

    def test_dream_null_text(self):
        resp = client.post("/api/v1/generate/dream", json={"text": None, "strength": 0.5})
        assert resp.status_code == 422

    def test_dream_missing_strength(self):
        """Missing strength uses default (0.3), so request succeeds."""
        resp = client.post("/api/v1/generate/dream", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["strength"] == 0.3

    def test_nightmare_missing_text_field(self):
        resp = client.post("/api/v1/generate/nightmare", json={"strength": 0.5})
        assert resp.status_code == 422

    def test_nightmare_null_text(self):
        resp = client.post("/api/v1/generate/nightmare", json={"text": None, "strength": 0.5})
        assert resp.status_code == 422

    def test_robustness_missing_text(self):
        resp = client.post("/api/v1/evaluate/robustness", json={"strengths": [0.2, 0.5]})
        assert resp.status_code == 422

    def test_robustness_null_strengths(self):
        resp = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": "hello", "strengths": None},
        )
        assert resp.status_code == 422

    def test_robustness_empty_strengths_list(self):
        resp = client.post(
            "/api/v1/evaluate/robustness",
            json={"text": "hello world test", "strengths": []},
        )
        # Might be 422 or 200 with empty results — either is acceptable
        assert resp.status_code in (200, 422)

    def test_dream_invalid_json_body(self):
        resp = client.post(
            "/api/v1/generate/dream",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_dream_strength_out_of_range(self):
        resp = client.post("/api/v1/generate/dream", json={"text": "test", "strength": 5.0})
        assert resp.status_code == 422

    def test_nightmare_strength_negative(self):
        resp = client.post("/api/v1/generate/nightmare", json={"text": "test", "strength": -1.0})
        assert resp.status_code == 422

    def test_dream_empty_body(self):
        resp = client.post(
            "/api/v1/generate/dream",
            json={},
        )
        assert resp.status_code == 422
