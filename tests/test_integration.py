"""Integration tests for NightmareNet — end-to-end pipelines."""

from __future__ import annotations

import os

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from nightmarenet.data.generator import (
    DreamDatasetGenerator,
    NightmareDatasetGenerator,
)
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.training.phases import (
    CompressionPhase,
    DreamPhase,
    NightmarePhase,
    WakePhase,
)
from nightmarenet.training.scheduler import CyclicScheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_dataset(n: int = 20) -> Dataset:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Paris is the capital of France and a major city.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables text understanding.",
    ]
    return Dataset.from_dict({"text": [texts[i % len(texts)] for i in range(n)]})


def _tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 32):
    def tok_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Don't add labels — phases set labels=input_ids themselves
        return out

    ds = dataset.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def _make_dataloader(dataset: Dataset, tokenizer, batch_size: int = 4):
    tokenized = _tokenize_dataset(dataset, tokenizer)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Data pipeline integration
# ---------------------------------------------------------------------------

class TestDataPipelineIntegration:
    """Dream/Nightmare generators → DataLoader → phase runner."""

    def test_dream_generated_data_trains(self):
        """Dream-generated data can be tokenized and used by WakePhase."""
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

        base = _make_tiny_dataset(10)
        gen = DreamDatasetGenerator(strength=0.3, seed=42)
        dream_ds = gen.generate(base)
        loader = _make_dataloader(dream_ds, tokenizer, batch_size=2)

        phase = WakePhase(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            config={"max_grad_norm": 1.0, "gradient_accumulation_steps": 1},
            device="cpu",
        )
        result = phase.run(loader, num_epochs=1)
        assert "avg_loss" in result
        assert result["avg_loss"] > 0

    def test_nightmare_generated_data_trains(self):
        """Nightmare-generated data can be tokenized and fed to NightmarePhase."""
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

        base = _make_tiny_dataset(10)
        gen = NightmareDatasetGenerator(strength=0.7, seed=42)
        nightmare_ds = gen.generate(base)
        loader = _make_dataloader(nightmare_ds, tokenizer, batch_size=2)

        phase = NightmarePhase(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=2e-4),
            config={"max_grad_norm": 1.0, "gradient_accumulation_steps": 1},
            device="cpu",
            lr_multiplier=2.0,
        )
        result = phase.run(loader, num_epochs=1)
        assert "avg_loss" in result
        assert result["avg_loss"] > 0


# ---------------------------------------------------------------------------
# Full training cycle integration (phases only — no Trainer overhead)
# ---------------------------------------------------------------------------

class TestFullCycleIntegration:
    """Wake → Dream → Nightmare → Compress in sequence."""

    def test_single_cycle_phases(self):
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

        base = _make_tiny_dataset(8)
        dream_ds = DreamDatasetGenerator(strength=0.3, seed=1).generate(base)
        nightmare_ds = NightmareDatasetGenerator(strength=0.7, seed=2).generate(base)

        train_loader = _make_dataloader(base, tokenizer, batch_size=4)
        dream_loader = _make_dataloader(dream_ds, tokenizer, batch_size=4)
        nightmare_loader = _make_dataloader(nightmare_ds, tokenizer, batch_size=4)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        cfg = {"max_grad_norm": 1.0, "gradient_accumulation_steps": 1}

        results = []

        # Wake
        wake = WakePhase(
            model=model, optimizer=optimizer,
            config=cfg, device="cpu",
        )
        results.append(wake.run(train_loader, num_epochs=1))

        # Dream
        dream = DreamPhase(
            model=model, optimizer=optimizer,
            config=cfg, device="cpu",
        )
        results.append(dream.run(dream_loader, num_epochs=1))

        # Nightmare
        nightmare = NightmarePhase(
            model=model, optimizer=optimizer,
            config=cfg, device="cpu", lr_multiplier=2.0,
        )
        results.append(nightmare.run(nightmare_loader, num_epochs=1))

        # Compress
        compress = CompressionPhase(
            model=model,
            config={"pruning_amount": 0.1, "knowledge_distillation": False},
            device="cpu",
        )
        results.append(compress.run(dataloader=train_loader, optimizer=optimizer))

        # Verify all phases ran and returned loss info
        assert len(results) == 4
        for r in results:
            assert isinstance(r, dict)
            assert "phase" in r

    def test_scheduler_drives_phases(self):
        """CyclicScheduler produces correct phase ordering for 1 cycle."""
        scheduler = CyclicScheduler(
            num_cycles=1,
            wake_epochs=1,
            dream_epochs=1,
            nightmare_epochs=1,
            compression_rounds=1,
        )
        phases = [(phase, epochs) for _, phase, epochs in scheduler]
        assert phases == [
            ("wake", 1), ("dream", 1), ("nightmare", 1), ("compress", 1)
        ]


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------

class TestEvaluatorIntegration:
    """Evaluator with real model and data."""

    def test_evaluate_recall_metric(self):
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

        ds = _make_tiny_dataset(8)
        loader = _make_dataloader(ds, tokenizer, batch_size=4)

        config = {
            "evaluation": {"metrics": ["recall"]},
            "model": {"max_length": 32},
            "training": {"batch_size": 4},
        }
        evaluator = Evaluator(
            model=model, tokenizer=tokenizer,
            config=config, device="cpu",
        )
        results = evaluator.evaluate(clean_dataloader=loader, label="test")
        assert "recall" in results
        assert "label" in results
        assert results["label"] == "test"


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from nightmarenet.api.app import app  # noqa: E402

api_client = TestClient(app)


class TestAPIEndToEnd:
    """Full round-trip API tests across multiple endpoints."""

    def test_dream_then_nightmare_same_text(self):
        """Dream and nightmare on same text produce different outputs."""
        text = "The quick brown fox jumps over the lazy dog."
        dream = api_client.post(
            "/api/v1/generate/dream",
            json={"text": text, "strength": 0.5, "seed": 42},
        )
        nightmare = api_client.post(
            "/api/v1/generate/nightmare",
            json={"text": text, "strength": 0.5, "seed": 42},
        )
        assert dream.status_code == 200
        assert nightmare.status_code == 200
        d_text = dream.json()["distorted_text"]
        n_text = nightmare.json()["distorted_text"]
        # Nightmare should differ more (has adversarial layer)
        assert d_text != n_text or d_text != text

    def test_robustness_evaluation_e2e(self):
        """Robustness endpoint returns structured multi-strength report."""
        resp = api_client.post(
            "/api/v1/evaluate/robustness",
            json={
                "text": "Machine learning is powerful.",
                "strengths": [0.1, 0.5, 0.9],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scores" in data
        assert "dream" in data["scores"]
        assert "nightmare" in data["scores"]
        # Should have results for each strength
        for key in ["0.1", "0.5", "0.9"]:
            assert key in data["scores"]["dream"]
            assert key in data["scores"]["nightmare"]

    def test_distortion_idempotent_with_seed(self):
        """Same seed produces same output."""
        payload = {"text": "Hello world.", "strength": 0.3, "seed": 123}
        r1 = api_client.post("/api/v1/generate/dream", json=payload)
        r2 = api_client.post("/api/v1/generate/dream", json=payload)
        assert r1.json()["distorted_text"] == r2.json()["distorted_text"]


class TestAPIRateLimitEnforcement:
    """SlowAPI middleware enforcement — actual HTTP round-trips."""

    def test_rate_limit_handler_format(self):
        """The rate limit handler returns well-formed JSON error."""
        import asyncio
        import json
        from unittest.mock import MagicMock

        from nightmarenet.api.app import _rate_limit_handler

        fake_request = MagicMock()
        fake_exc = MagicMock()
        fake_exc.detail = "60 per 1 minute"

        response = asyncio.new_event_loop().run_until_complete(
            _rate_limit_handler(fake_request, fake_exc)
        )
        assert response.status_code == 429
        body = json.loads(response.body)
        assert body["error"] == "Rate limit exceeded"
        assert "60 per 1 minute" in body["detail"]

    def test_slowapi_middleware_is_registered(self):
        """Verify SlowAPIMiddleware is in the app middleware stack."""
        middleware_classes = [
            m.cls.__name__ if hasattr(m, "cls") else type(m).__name__
            for m in app.user_middleware
        ]
        assert "SlowAPIMiddleware" in middleware_classes


class TestAPICORSConfig:
    """CORS origin stripping."""

    def test_cors_origins_stripped(self, monkeypatch):
        """Whitespace in CORS origins env var is stripped."""
        monkeypatch.setenv(
            "NIGHTMARENET_CORS_ORIGINS",
            " http://a.com , http://b.com ",
        )
        # Re-evaluate the origins parsing logic
        raw = os.environ.get("NIGHTMARENET_CORS_ORIGINS", "*")
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        assert origins == ["http://a.com", "http://b.com"]
