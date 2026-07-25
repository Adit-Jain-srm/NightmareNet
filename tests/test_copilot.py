"""Tests for the copilot endpoint.

Covers the heuristic backend (default, no LLM API keys required). The LLM
backends are tested by integration; here we verify the public contract
between the dock and the API so the wiring stays honest.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from nightmarenet.api.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_llm_keys(monkeypatch):
    """Ensure the heuristic backend is selected for every test."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE body into a list of event dicts."""
    events: list[dict] = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


class TestCopilotNonStream:
    """The JSON (non-stream) response shape is the contract anchor."""

    def test_returns_200_with_expected_shape(self):
        response = client.post(
            "/api/v1/copilot/ask",
            json={
                "question": "What should I do next?",
                "section": "command-center",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert set(data.keys()) == {"answer", "suggestions", "model"}
        assert isinstance(data["answer"], str) and data["answer"]
        assert isinstance(data["suggestions"], list)
        assert data["model"] == "heuristic"

        for sug in data["suggestions"]:
            assert set(sug.keys()) == {"label", "action", "detail"}
            assert isinstance(sug["label"], str)
            assert isinstance(sug["action"], str)
            assert isinstance(sug["detail"], str)

    def test_context_augments_answer(self):
        response = client.post(
            "/api/v1/copilot/ask",
            json={
                "question": "Where am I weakest?",
                "section": "robustness",
                "stream": False,
                "context": {"last_run_robustness": 0.66, "recent_runs": 3},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "66%" in data["answer"]
        assert "3" in data["answer"]

    def test_unknown_section_falls_back_to_command_center(self):
        response = client.post(
            "/api/v1/copilot/ask",
            json={
                "question": "Hello",
                "section": "made-up-section",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "heuristic"
        # command-center hint mentions "Welcome back."
        assert "Welcome back" in data["answer"]

    def test_validation_rejects_empty_question(self):
        response = client.post(
            "/api/v1/copilot/ask",
            json={"question": "", "section": "command-center", "stream": False},
        )
        assert response.status_code == 422


class TestCopilotStream:
    """SSE stream emits the same shape, terminated by a `done` event."""

    def test_stream_emits_tokens_and_done(self):
        with client.stream(
            "POST",
            "/api/v1/copilot/ask",
            json={
                "question": "How is my model doing?",
                "section": "metrics",
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = response.read().decode("utf-8")

        events = _parse_sse(body)
        assert len(events) >= 2, f"Expected at least 1 token + done; got: {events}"

        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]

        assert token_events, "Expected at least one token event"
        assert len(done_events) == 1, "Expected exactly one done event"

        done = done_events[0]
        assert done["model"] == "heuristic"
        assert isinstance(done["suggestions"], list)
        for sug in done["suggestions"]:
            assert {"label", "action", "detail"} <= set(sug.keys())

        # Reconstructed answer should be non-empty.
        reconstructed = "".join(e["token"] for e in token_events).strip()
        assert reconstructed
