"""NightmareNet AI copilot endpoint.

Powers the dashboard's "Ask NightmareNet" dock with a streaming
context-aware answer. The endpoint auto-selects a backend at request time:

1. **LLM mode** — OpenAI, Anthropic, or Azure OpenAI when the corresponding
   ``*_API_KEY`` env var is set *and* the SDK is importable. Tokens stream
   via Server-Sent Events.
2. **Heuristic mode** — deterministic context-aware answer keyed by
   dashboard section. Used by default and when LLM mode fails. Streams a
   word-by-word approximation through the same SSE shape so the frontend
   never has to branch on backend.

Both modes emit the same event vocabulary:

* ``{"token": "..."}`` — one token delta
* ``{"done": true, "suggestions": [...], "model": "<id>"}`` — final event

Per :file:`CLAUDE.md`: this file deliberately does NOT use
``from __future__ import annotations`` because Pydantic v2 + FastAPI
``Body(...)`` is incompatible with PEP 563 deferred evaluation.

Mounted by :func:`register_copilot_routes` from :mod:`nightmarenet.api.app`.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are NightmareNet's AI copilot — an expert in adversarial robustness, "
    "the 4-phase Wake/Dream/Nightmare/Compress training cycle, and model "
    "hardening. You have deep knowledge of:\n"
    "- Distortion engines (character, word, semantic, adversarial, learned-attention)\n"
    "- The +14.49% relative robustness improvement benchmark (SST-2, DistilBERT, seed 42)\n"
    "- Pipeline lifecycle: ingest → optimize (Adaption Labs) → prepare → train → evaluate\n"
    "- Supported configs: wake_epochs, dream/nightmare strength, compression ratio, batch size\n"
    "- Hardware constraints: RTX 3050 Ti 4GB, FP16 AMP, gradient checkpointing, batch 4-16\n"
    "\n"
    "Rules:\n"
    "- Answer in 3 sentences max.\n"
    "- Suggest 1-2 concrete next dashboard actions.\n"
    "- Never invent metrics; reference real numbers if available in context.\n"
    "- If the user's question is about config tuning, reference the benchmark results.\n"
    "- If dataset optimization is relevant, mention Adaption Labs integration."
)

VALID_SECTIONS = frozenset(
    {
        "command-center",
        "experiments",
        "run-detail",
        "phases",
        "metrics",
        "robustness",
        "compare",
        "distortions",
        "audit",
        "benchmarks",
        "ci",
        "settings",
        "data-quality",
    }
)

DEFAULT_SECTION = "command-center"

# Lifted from the frontend dock's buildAnswer() heuristic — single source of
# truth on the server so the dock can stay thin once it talks to the API.
CONTEXTUAL_HINTS: Dict[str, Dict[str, Any]] = {
    "command-center": {
        "hint": (
            "Welcome back. Your last cycle improved robustness by +13.6% — "
            "keep going with the next benchmark or stress-test an unseen attack."
        ),
        "next": [
            {
                "label": "Run standard benchmark",
                "action": "benchmarks",
                "detail": "DistilBERT \u00b7 SST-2 \u00b7 4-phase cycle",
            },
            {
                "label": "Stress test current model",
                "action": "distortions",
                "detail": "Sweep dream + nightmare 0.1-0.9",
            },
        ],
    },
    "experiments": {
        "hint": (
            "Compare your two most recent runs side-by-side to see which "
            "configuration is converging fastest."
        ),
        "next": [
            {
                "label": "Open Model Comparison",
                "action": "compare",
                "detail": "A/B overlay of latest two runs",
            }
        ],
    },
    "run-detail": {
        "hint": (
            "This run is in the Nightmare phase. Open the radar to see which "
            "attack family it's least robust against — that's where the next "
            "cycle should focus."
        ),
        "next": [
            {
                "label": "Inspect robustness radar",
                "action": "robustness",
                "detail": "5-axis weakness map",
            }
        ],
    },
    "phases": {
        "hint": (
            "Tune the nightmare strength schedule and re-run the cycle. Most "
            "models gain another 4-7% robustness when the nightmare phase is "
            "extended by one epoch."
        ),
        "next": [
            {
                "label": "Open Metrics",
                "action": "metrics",
                "detail": "Watch loss + robustness curves",
            }
        ],
    },
    "metrics": {
        "hint": (
            "Loss and robustness curves are the fastest signal of overfitting. "
            "If robustness flatlines but loss keeps dropping, drop dream "
            "strength by 0.1."
        ),
        "next": [
            {
                "label": "Compare two runs",
                "action": "compare",
                "detail": "Pick the latest pair for A/B overlay",
            }
        ],
    },
    "robustness": {
        "hint": (
            "Your weakest axis is semantic distortion at high strength. "
            "Schedule a Nightmare-heavy cycle to harden it."
        ),
        "next": [
            {
                "label": "Open Phase Visualizer",
                "action": "phases",
                "detail": "Tune nightmare strength schedule",
            }
        ],
    },
    "compare": {
        "hint": (
            "Diff the two runs by phase. The earliest cycle where one model "
            "pulls ahead usually points at the right hyperparameter change."
        ),
        "next": [
            {
                "label": "Open Robustness Radar",
                "action": "robustness",
                "detail": "Per-axis weakness map",
            }
        ],
    },
    "distortions": {
        "hint": (
            "Try the same input across strengths 0.1, 0.5, and 0.9 to see how "
            "nightmare distortion escalates — and where your model's "
            "decision boundary breaks."
        ),
        "next": [
            {
                "label": "Watch live metrics",
                "action": "metrics",
                "detail": "Loss + robustness curves",
            }
        ],
    },
    "audit": {
        "hint": (
            "Filter by error events to triage failures faster — most "
            "regressions cluster in the first two cycles after a config change."
        ),
        "next": [
            {
                "label": "Open CI integration",
                "action": "ci",
                "detail": "Wire the robustness gate into pull requests",
            }
        ],
    },
    "benchmarks": {
        "hint": (
            "Standard benchmark is DistilBERT on SST-2 with a 4-phase cycle. "
            "Drop the threshold to your current avg distorted accuracy minus "
            "0.02 to gate PRs without false alarms."
        ),
        "next": [
            {
                "label": "Open Run Detail",
                "action": "run-detail",
                "detail": "Inspect the latest benchmark cycle",
            }
        ],
    },
    "ci": {
        "hint": (
            "The robustness-check Action is wired. Set your threshold to your "
            "model's current avg distorted accuracy minus 0.02 to catch "
            "regressions without false alarms."
        ),
        "next": [
            {
                "label": "Open Settings",
                "action": "settings",
                "detail": "Manage API keys + thresholds",
            }
        ],
    },
    "settings": {
        "hint": (
            "Rotate API keys at least quarterly and configure a robustness "
            "threshold per environment. The CI gate reads the same threshold."
        ),
        "next": [
            {
                "label": "Open CI Integration",
                "action": "ci",
                "detail": "Configure the robustness gate",
            }
        ],
    },
    "data-quality": {
        "hint": (
            "Adaption Labs optimization can improve dataset quality before training — "
            "deduplication, noise filtering, and curriculum scoring typically boost "
            "robustness by 3-8% with no architecture changes."
        ),
        "next": [
            {
                "label": "Run Data Optimization",
                "action": "data-quality",
                "detail": "Adaption Labs \u00b7 denoise + deduplicate + score",
            },
            {
                "label": "View Quality Metrics",
                "action": "metrics",
                "detail": "Dataset health: duplicates, noise ratio, coverage",
            },
        ],
    },
}

_DEFAULT_HINT_ENTRY: Dict[str, Any] = {
    "hint": "Tip: press Cmd+K to jump anywhere, or ? to see every shortcut.",
    "next": [
        {
            "label": "Open Command Palette",
            "action": "command-center",
            "detail": "Cmd+K opens every action",
        }
    ],
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CopilotSuggestion(BaseModel):
    """A single suggested next action surfaced by the copilot."""

    label: str
    action: str
    detail: str


class CopilotAskRequest(BaseModel):
    """Request body for the copilot ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Free-text question from the dashboard dock.",
    )
    section: str = Field(
        default=DEFAULT_SECTION,
        description="Current dashboard section the user is viewing.",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Free-form dashboard context (e.g. last_run_robustness).",
    )
    stream: bool = Field(
        default=True,
        description="If true, respond with SSE; otherwise return a single JSON object.",
    )


class CopilotAskResponse(BaseModel):
    """Non-streaming response shape; matches the final SSE event payload."""

    answer: str
    suggestions: List[CopilotSuggestion]
    model: str


# Module-level Body singleton to dodge ruff B008 the same way app.py does.
_COPILOT_BODY = Body(...)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _section_or_default(section: str) -> str:
    return section if section in VALID_SECTIONS else DEFAULT_SECTION


def _section_entry(section: str) -> Dict[str, Any]:
    return CONTEXTUAL_HINTS.get(_section_or_default(section), _DEFAULT_HINT_ENTRY)


def _detect_llm_backend() -> Optional[str]:
    """Return the name of an available LLM backend, or ``None`` for heuristic.

    Order: Azure OpenAI > OpenAI > Anthropic.
    SDK availability is checked lazily so the OSS install path never requires
    these packages.
    """
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        try:
            import openai  # noqa: F401

            return "azure"
        except ImportError:
            logger.debug("AZURE_OPENAI_API_KEY set but `openai` SDK not importable")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401

            return "openai"
        except ImportError:
            logger.debug("OPENAI_API_KEY set but `openai` SDK not importable")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401

            return "anthropic"
        except ImportError:
            logger.debug("ANTHROPIC_API_KEY set but `anthropic` SDK not importable")
    return None


def _build_user_prompt(question: str, section: str, context: Optional[Dict[str, Any]]) -> str:
    """Render the user-side LLM prompt with enriched runtime context."""
    ctx = dict(context) if context else {}

    # Auto-inject live metrics from benchmark results if available
    try:
        import pathlib

        bench_base = pathlib.Path(os.environ.get("NIGHTMARENET_RESULTS_DIR", "results"))
        bench_path = bench_base / "gpu_benchmark.json"
        if bench_path.exists():
            bench = json.loads(bench_path.read_text(encoding="utf-8"))
            comparison = bench.get("comparison", {})
            if comparison and "robustness_improvement_pct" not in ctx:
                ctx["benchmark_robustness_improvement_pct"] = comparison.get(
                    "robustness_improvement_pct"
                )
                ctx["benchmark_clean_delta"] = comparison.get("clean_delta")
                ctx["benchmark_avg_distorted_delta"] = comparison.get("avg_distorted_delta")
    except Exception:
        pass

    # Auto-inject available distortion engines
    try:
        from nightmarenet.distortions.registry import get_registry

        reg = get_registry()
        ctx["available_distortion_engines"] = list(reg._engines.keys())
    except Exception:
        pass

    # Auto-inject pipeline runner state
    try:
        from nightmarenet.pipeline_runner import list_runners

        runners = list_runners()
        active = [r for r in runners if r.get("status") in ("running", "preparing")]
        if active:
            ctx["active_runs"] = len(active)
    except Exception:
        pass

    safe_context = json.dumps(ctx, ensure_ascii=False, sort_keys=True, default=str)
    section_hint = _section_entry(section)["hint"]
    temperature = float(os.environ.get("NIGHTMARENET_COPILOT_TEMPERATURE", "0.4"))
    return (
        f"Dashboard section: {section}\n"
        f"Section context hint: {section_hint}\n"
        f"Dashboard runtime context: {safe_context}\n"
        f"Temperature setting: {temperature}\n"
        f"User question: {question}"
    )


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------


def _heuristic_answer_text(section: str, question: str, context: Optional[Dict[str, Any]]) -> str:
    """Deterministic, context-aware answer string for the heuristic backend."""
    entry = _section_entry(section)
    parts: List[str] = []

    if context:
        rob = context.get("last_run_robustness")
        if isinstance(rob, (int, float)):
            parts.append(f"Your last run sits at {float(rob):.0%} robustness.")
        recent = context.get("recent_runs")
        if isinstance(recent, int) and recent > 0:
            parts.append(f"You've logged {recent} run(s) recently.")

    parts.append(entry["hint"])

    q = (question or "").strip().lower()
    if q:
        for sug in entry["next"]:
            label = str(sug.get("label", "")).lower()
            if label and any(tok and tok in label for tok in q.split()):
                parts.append(f"Closest action: {sug['label']} \u2014 {sug['detail']}.")
                break

    return " ".join(parts).strip()


def _suggestions_for(section: str) -> List[Dict[str, str]]:
    entry = _section_entry(section)
    return [dict(s) for s in entry["next"]]


# ---------------------------------------------------------------------------
# LLM streaming adapters (all imports guarded)
# ---------------------------------------------------------------------------


def _openai_model() -> str:
    return os.environ.get("NIGHTMARENET_COPILOT_MODEL", "gpt-4o-mini")


def _anthropic_model() -> str:
    return os.environ.get("NIGHTMARENET_COPILOT_MODEL", "claude-3-5-haiku-latest")


def _azure_deployment() -> str:
    return os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT",
        os.environ.get("NIGHTMARENET_COPILOT_MODEL", "gpt-4o-mini"),
    )


async def _stream_openai(
    question: str, section: str, context: Optional[Dict[str, Any]]
) -> AsyncIterator[str]:
    import openai  # type: ignore[import-not-found]

    client = openai.AsyncOpenAI()
    prompt = _build_user_prompt(question, section, context)
    stream = await client.chat.completions.create(
        model=_openai_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=400,
        temperature=0.4,
    )
    async for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            continue
        if delta:
            yield delta


async def _stream_anthropic(
    question: str, section: str, context: Optional[Dict[str, Any]]
) -> AsyncIterator[str]:
    import anthropic  # type: ignore[import-not-found]

    client = anthropic.AsyncAnthropic()
    prompt = _build_user_prompt(question, section, context)
    async with client.messages.stream(
        model=_anthropic_model(),
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    ) as stream:
        async for text in stream.text_stream:
            if text:
                yield text


async def _stream_azure(
    question: str, section: str, context: Optional[Dict[str, Any]]
) -> AsyncIterator[str]:
    import openai  # type: ignore[import-not-found]

    client = openai.AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )
    prompt = _build_user_prompt(question, section, context)
    stream = await client.chat.completions.create(
        model=_azure_deployment(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=400,
        temperature=0.4,
    )
    async for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            continue
        if delta:
            yield delta


_LLM_DISPATCH = {
    "openai": (_stream_openai, lambda: f"openai:{_openai_model()}"),
    "anthropic": (_stream_anthropic, lambda: f"anthropic:{_anthropic_model()}"),
    "azure": (_stream_azure, lambda: f"azure:{_azure_deployment()}"),
}


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------


def _sse(event: Dict[str, Any]) -> str:
    """Serialise an event dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _heuristic_token_stream(answer: str) -> AsyncIterator[str]:
    """Word-by-word approximation so the dock feels live in heuristic mode."""
    pieces = answer.split(" ")
    for idx, word in enumerate(pieces):
        if not word:
            continue
        token = word if idx == 0 else " " + word
        yield token
        # Yield control so the SSE flushes incrementally.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


async def _build_sse_stream(body: CopilotAskRequest) -> AsyncIterator[str]:
    section = _section_or_default(body.section)
    suggestions = _suggestions_for(section)
    backend = _detect_llm_backend()

    if backend in _LLM_DISPATCH:
        stream_fn, model_id_fn = _LLM_DISPATCH[backend]
        start_time = time.time()
        try:
            async for token in stream_fn(body.question, section, body.context):
                yield _sse({"token": token})
            latency_ms = round((time.time() - start_time) * 1000, 1)
            yield _sse(
                {
                    "done": True,
                    "suggestions": suggestions,
                    "model": model_id_fn(),
                    "latency_ms": latency_ms,
                }
            )
            return
        except Exception as exc:
            logger.warning(
                "Copilot LLM backend '%s' failed, falling back to heuristic: %s",
                backend,
                exc,
            )

    start_time = time.time()
    answer = _heuristic_answer_text(section, body.question, body.context)
    async for token in _heuristic_token_stream(answer):
        yield _sse({"token": token})
    latency_ms = round((time.time() - start_time) * 1000, 1)
    yield _sse(
        {
            "done": True,
            "suggestions": suggestions,
            "model": "heuristic",
            "latency_ms": latency_ms,
        }
    )


async def _build_non_stream_response(body: CopilotAskRequest) -> Dict[str, Any]:
    section = _section_or_default(body.section)
    suggestions = _suggestions_for(section)
    backend = _detect_llm_backend()

    if backend in _LLM_DISPATCH:
        stream_fn, model_id_fn = _LLM_DISPATCH[backend]
        try:
            chunks: List[str] = []
            async for token in stream_fn(body.question, section, body.context):
                chunks.append(token)
            answer = "".join(chunks).strip()
            if answer:
                return {
                    "answer": answer,
                    "suggestions": suggestions,
                    "model": model_id_fn(),
                }
        except Exception as exc:
            logger.warning(
                "Copilot LLM backend '%s' failed (non-stream), falling back to heuristic: %s",
                backend,
                exc,
            )

    answer = _heuristic_answer_text(section, body.question, body.context)
    return {
        "answer": answer,
        "suggestions": suggestions,
        "model": "heuristic",
    }


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def register_copilot_routes(app: FastAPI, limiter: Limiter) -> None:
    """Mount the copilot endpoint onto ``app`` with shared rate limiting.

    Called once from :mod:`nightmarenet.api.app` after the limiter is built.
    Using a registration function (instead of an ``APIRouter`` at import time)
    keeps the limiter wiring explicit and avoids circular imports.
    """

    @app.post(
        "/api/v1/copilot/ask",
        tags=["copilot"],
        summary="Ask the NightmareNet copilot",
        responses={
            200: {
                "description": ("Either an SSE stream (when stream=true) or a JSON answer."),
                "content": {
                    "text/event-stream": {
                        "schema": {
                            "type": "string",
                            "description": (
                                "Sequence of `data: {...}` lines. Each line is "
                                "either a `{token}` event or the terminal "
                                "`{done, suggestions, model}` event."
                            ),
                        }
                    },
                    "application/json": {
                        "schema": CopilotAskResponse.model_json_schema(),
                    },
                },
            }
        },
    )
    @limiter.limit("20/minute")
    async def copilot_ask(  # noqa: D401  (FastAPI handler)
        request: Request, body: CopilotAskRequest = _COPILOT_BODY
    ) -> Any:
        """Stream or return a context-aware copilot answer."""
        if body.stream:
            return StreamingResponse(
                _build_sse_stream(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )
        result = await _build_non_stream_response(body)
        return result
