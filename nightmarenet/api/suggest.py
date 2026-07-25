"""NightmareNet AI-powered config suggestion endpoint.

Analyzes a training configuration plus optional last-run metrics and returns
actionable hyperparameter improvement suggestions. Supports multi-backend
failover: Azure OpenAI → Heuristic.

Features:
- In-memory TTL cache (60s) to avoid duplicate LLM calls for identical configs
- JSON parse hardening (markdown fences, trailing commas, partial JSON)
- 15s timeout on LLM calls with graceful fallback to heuristic
- Input validation for impossible config values
- Response metadata (latency_ms, tokens_used)

Per :file:`CLAUDE.md`: this file deliberately does NOT use
``from __future__ import annotations`` because Pydantic v2 + FastAPI
``Body(...)`` is incompatible with PEP 563 deferred evaluation.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter

logger = logging.getLogger(__name__)

_SUGGEST_BODY = Body(...)

_LLM_TIMEOUT_SECONDS = 15.0

# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 60.0
_cache: Dict[str, Tuple[float, List[Dict[str, Any]], str, Optional[int]]] = {}


def _cache_key(
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    hardware: Optional[str],
) -> str:
    raw = json.dumps(
        {"config": config, "metrics": metrics, "hardware": hardware},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(
    key: str,
) -> Optional[Tuple[List[Dict[str, Any]], str, Optional[int]]]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, suggestions, model, tokens = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return suggestions, model, tokens


def _cache_set(
    key: str,
    suggestions: List[Dict[str, Any]],
    model: str,
    tokens: Optional[int],
) -> None:
    _cache[key] = (time.time(), suggestions, model, tokens)
    if len(_cache) > 256:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest_key, None)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ConfigSuggestion(BaseModel):
    """A single config suggestion with current/suggested values and reasoning."""

    param: str = Field(..., description="Parameter path (e.g. 'batch_size')")
    current: Any = Field(..., description="Current value in the config")
    suggested: Any = Field(..., description="Suggested new value")
    reason: str = Field(..., description="Why this change helps")


class SuggestConfigRequest(BaseModel):
    """Request body for the config suggestion endpoint."""

    current_config: Dict[str, Any] = Field(..., description="Training YAML as a dict")
    last_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last run metrics (clean_accuracy, avg_distorted_accuracy, etc.)",
    )
    hardware: Optional[str] = Field(
        default=None,
        description="Hardware description (e.g. 'RTX 3050 Ti 4GB')",
    )

    @field_validator("current_config")
    @classmethod
    def validate_config_values(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Reject configs with impossible values."""
        training = v.get("training", v)

        lr = training.get("learning_rate")
        if lr is not None and (not isinstance(lr, (int, float)) or lr <= 0):
            raise ValueError(f"learning_rate must be positive, got {lr}")

        batch_size = training.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        num_cycles = training.get("num_cycles")
        if num_cycles is not None and (not isinstance(num_cycles, int) or num_cycles < 1):
            raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")

        epochs = training.get("wake_epochs")
        if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
            raise ValueError(f"wake_epochs must be >= 1, got {epochs}")

        distortion = v.get("distortion", v)
        for strength_key in ("dream_strength", "nightmare_strength"):
            val = distortion.get(strength_key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0.0 or val > 1.0):
                raise ValueError(f"{strength_key} must be in [0.0, 1.0], got {val}")

        return v


class SuggestConfigResponse(BaseModel):
    """Response from the config suggestion endpoint."""

    suggestions: List[ConfigSuggestion]
    model: str = Field(..., description="Backend used: LLM model id or 'heuristic'")
    latency_ms: Optional[float] = Field(
        default=None, description="End-to-end latency in milliseconds"
    )
    tokens_used: Optional[int] = Field(
        default=None, description="LLM tokens consumed (None for heuristic)"
    )


# ---------------------------------------------------------------------------
# JSON parse hardening
# ---------------------------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def _parse_llm_json(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output to extract suggestions list, handling common issues."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    text = _TRAILING_COMMA_RE.sub(r"\1", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        bracket_start = text.find("{")
        bracket_end = text.rfind("}")
        if bracket_start >= 0 and bracket_end > bracket_start:
            candidate = text[bracket_start : bracket_end + 1]
            candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                array_start = text.find("[")
                array_end = text.rfind("]")
                if array_start >= 0 and array_end > array_start:
                    candidate = text[array_start : array_end + 1]
                    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        return []
                else:
                    return []
        else:
            return []

    if isinstance(parsed, dict) and "suggestions" in parsed:
        result = parsed["suggestions"]
        if isinstance(result, list):
            return result
    if isinstance(parsed, list):
        return parsed
    return []


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_azure_available() -> bool:
    """Check if Azure OpenAI is configured and importable."""
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        return False
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return False
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def _pick_llm_backend() -> Optional[str]:
    if _detect_azure_available():
        return "azure"
    return None


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are NightmareNet's config optimizer. Given a training config and "
    "optional metrics from the last run, suggest 3-5 hyperparameter improvements.\n\n"
    "Rules:\n"
    '- Return ONLY valid JSON: {"suggestions": [{"param": str, "current": value, '
    '"suggested": value, "reason": str}, ...]}\n'
    "- Focus on measurable improvements to robustness, convergence, or efficiency.\n"
    "- Consider hardware constraints if provided.\n"
    "- Each reason should be 1-2 sentences max.\n"
    "- Param should be a dot-separated path like 'training.batch_size'."
)


def _azure_deployment() -> str:
    return os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT",
        os.environ.get("NIGHTMARENET_COPILOT_MODEL", "gpt-4o"),
    )


async def _suggest_via_azure(
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    hardware: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Call Azure OpenAI to get structured suggestions."""
    import asyncio

    import openai  # type: ignore[import-not-found]

    client = openai.AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )

    user_content = json.dumps(
        {"current_config": config, "last_metrics": metrics, "hardware": hardware},
        indent=2,
        default=str,
    )

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=_azure_deployment(),
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=800,
            temperature=0.3,
        ),
        timeout=_LLM_TIMEOUT_SECONDS,
    )

    text = (response.choices[0].message.content or "").strip()
    tokens_used: Optional[int] = None
    if response.usage:
        tokens_used = response.usage.total_tokens

    return _parse_llm_json(text), tokens_used


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------

_VRAM_KEYWORDS = {
    "3050": 4,
    "3060": 12,
    "3070": 8,
    "3080": 10,
    "3090": 24,
    "4060": 8,
    "4070": 12,
    "4080": 16,
    "4090": 24,
    "a100": 80,
    "4gb": 4,
    "6gb": 6,
    "8gb": 8,
    "12gb": 12,
    "16gb": 16,
    "24gb": 24,
}


def _estimate_vram(hardware: Optional[str]) -> Optional[int]:
    """Rough VRAM estimate from hardware string."""
    if not hardware:
        return None
    hw_lower = hardware.lower()
    for key, vram in _VRAM_KEYWORDS.items():
        if key in hw_lower:
            return vram
    return None


def _heuristic_suggestions(
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    hardware: Optional[str],
) -> List[Dict[str, Any]]:
    """Generate rule-based suggestions from config + metrics."""
    suggestions: List[Dict[str, Any]] = []
    vram = _estimate_vram(hardware)

    training = config.get("training", config)
    distortion = config.get("distortion", config)

    batch_size = training.get("batch_size", distortion.get("batch_size"))
    dream_strength = distortion.get("dream_strength", training.get("dream_strength"))
    nightmare_strength = distortion.get("nightmare_strength", training.get("nightmare_strength"))
    learning_rate = training.get("learning_rate")
    num_cycles = training.get("num_cycles")

    # Rule 1: batch size vs VRAM (low VRAM)
    if batch_size is not None and vram is not None and vram < 6 and batch_size > 8:
        suggestions.append(
            {
                "param": "training.batch_size",
                "current": batch_size,
                "suggested": 4,
                "reason": (
                    f"With ~{vram}GB VRAM, batch_size {batch_size} risks OOM. "
                    "Use 4 with gradient accumulation for effective larger batches."
                ),
            }
        )

    # Rule 2: robustness drop too high
    if metrics:
        rob_drop = metrics.get("robustness_drop")
        if rob_drop is not None and rob_drop > 0.15:
            current_ns = nightmare_strength or 0.7
            suggested_ns = min(current_ns + 0.1, 0.95)
            suggestions.append(
                {
                    "param": "distortion.nightmare_strength",
                    "current": current_ns,
                    "suggested": round(suggested_ns, 2),
                    "reason": (
                        f"Robustness drop is {rob_drop:.1%} — increasing nightmare "
                        "strength forces the model to handle harder adversarial examples."
                    ),
                }
            )

    # Rule 3: dream strength too high
    if dream_strength is not None and dream_strength >= 0.5:
        suggestions.append(
            {
                "param": "distortion.dream_strength",
                "current": dream_strength,
                "suggested": 0.25,
                "reason": (
                    "Dream strength >= 0.5 is unusually aggressive. "
                    "Values of 0.2-0.3 give gentle generalization without degrading clean accuracy."
                ),
            }
        )

    # Rule 4: too few cycles
    if num_cycles is not None and num_cycles < 3:
        suggestions.append(
            {
                "param": "training.num_cycles",
                "current": num_cycles,
                "suggested": 3,
                "reason": (
                    "Fewer than 3 Wake/Dream/Nightmare/Compress cycles rarely "
                    "converges. 3-5 cycles is the sweet spot."
                ),
            }
        )

    # Rule 5: learning rate too high for fine-tuning
    if learning_rate is not None and learning_rate > 5e-4:
        suggestions.append(
            {
                "param": "training.learning_rate",
                "current": learning_rate,
                "suggested": 2e-5,
                "reason": (
                    "Learning rate > 5e-4 often causes catastrophic forgetting in "
                    "fine-tuning. 2e-5 is safer for pretrained transformers."
                ),
            }
        )

    # Rule 6: batch size too large for VRAM 4-8GB range
    if batch_size is not None and vram is not None and 4 <= vram <= 8 and batch_size > 16:
        if not any(s["param"] == "training.batch_size" for s in suggestions):
            suggestions.append(
                {
                    "param": "training.batch_size",
                    "current": batch_size,
                    "suggested": 8,
                    "reason": (
                        f"Batch size {batch_size} is large for {vram}GB VRAM. "
                        "Reducing to 8 prevents OOM and maintains throughput with "
                        "gradient accumulation steps."
                    ),
                }
            )

    # Rule 7: low clean accuracy suggests model underfitting
    if metrics:
        clean_acc = metrics.get("clean_accuracy")
        if clean_acc is not None and clean_acc < 0.7:
            wake_epochs = training.get("wake_epochs", 1)
            suggestions.append(
                {
                    "param": "training.wake_epochs",
                    "current": wake_epochs,
                    "suggested": max(wake_epochs + 1, 3),
                    "reason": (
                        f"Clean accuracy is only {clean_acc:.1%} — the model may be "
                        "underfitting. Increase wake epochs for more supervised learning."
                    ),
                }
            )

    # Rule 8: mixed precision recommendation
    fp16 = training.get("fp16", training.get("mixed_precision"))
    if fp16 is None or fp16 is False:
        if vram is not None and vram <= 16:
            suggestions.append(
                {
                    "param": "training.fp16",
                    "current": fp16 if fp16 is not None else "not set",
                    "suggested": True,
                    "reason": (
                        "Mixed precision (FP16) halves memory usage and speeds up "
                        "training 1.5-2x on modern GPUs with minimal accuracy loss."
                    ),
                }
            )

    # Rule 9: gradient accumulation steps calculation
    if batch_size is not None and batch_size <= 4:
        grad_accum = training.get("gradient_accumulation_steps", 1)
        target_effective = 16
        suggested_accum = max(target_effective // batch_size, 2)
        if grad_accum < suggested_accum:
            suggestions.append(
                {
                    "param": "training.gradient_accumulation_steps",
                    "current": grad_accum,
                    "suggested": suggested_accum,
                    "reason": (
                        f"With batch_size={batch_size}, accumulating {suggested_accum} "
                        f"steps gives effective batch of {batch_size * suggested_accum}, "
                        "improving gradient stability without extra memory."
                    ),
                }
            )

    # Rule 10: warmup ratio suggestion
    warmup = training.get("warmup_ratio", training.get("warmup_steps"))
    if warmup is None or warmup == 0:
        suggestions.append(
            {
                "param": "training.warmup_ratio",
                "current": warmup if warmup is not None else "not set",
                "suggested": 0.06,
                "reason": (
                    "A 6% warmup prevents early training instability and reduces "
                    "the risk of divergence in the first few steps."
                ),
            }
        )

    # Rule 11: weight decay check
    weight_decay = training.get("weight_decay")
    if weight_decay is not None and weight_decay > 0.1:
        suggestions.append(
            {
                "param": "training.weight_decay",
                "current": weight_decay,
                "suggested": 0.01,
                "reason": (
                    f"Weight decay of {weight_decay} is aggressive and can "
                    "under-parameterize the model. 0.01 is standard for AdamW fine-tuning."
                ),
            }
        )
    elif weight_decay is None and learning_rate is not None:
        suggestions.append(
            {
                "param": "training.weight_decay",
                "current": "not set",
                "suggested": 0.01,
                "reason": (
                    "Adding weight_decay=0.01 with AdamW provides regularization "
                    "that improves generalization without significant compute cost."
                ),
            }
        )

    return suggestions[:5]


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def register_suggest_routes(app: FastAPI, limiter: Limiter) -> None:
    """Mount the config suggestion endpoint onto ``app``."""

    @app.post(
        "/api/v1/suggest/config",
        response_model=SuggestConfigResponse,
        tags=["suggest"],
        summary="AI-powered config suggestions",
    )
    @limiter.limit("10/minute")
    async def suggest_config(
        request: Request,
        body: SuggestConfigRequest = _SUGGEST_BODY,
    ) -> SuggestConfigResponse:
        """Suggest hyperparameter improvements based on current config and metrics."""
        start = time.time()

        cache_k = _cache_key(body.current_config, body.last_metrics, body.hardware)
        cached = _cache_get(cache_k)
        if cached is not None:
            raw_suggestions, model, tokens = cached
            latency_ms = round((time.time() - start) * 1000, 1)
            suggestions = [
                ConfigSuggestion(
                    param=s.get("param", "unknown"),
                    current=s.get("current"),
                    suggested=s.get("suggested"),
                    reason=s.get("reason", ""),
                )
                for s in raw_suggestions
                if isinstance(s, dict)
            ]
            return SuggestConfigResponse(
                suggestions=suggestions,
                model=model,
                latency_ms=latency_ms,
                tokens_used=tokens,
            )

        backend = _pick_llm_backend()
        tokens_used: Optional[int] = None

        if backend == "azure":
            try:
                raw, tokens_used = await _suggest_via_azure(
                    body.current_config, body.last_metrics, body.hardware
                )
                if raw:
                    model_label = f"azure:{_azure_deployment()}"
                    _cache_set(cache_k, raw, model_label, tokens_used)
                    latency_ms = round((time.time() - start) * 1000, 1)
                    suggestions = [
                        ConfigSuggestion(
                            param=s.get("param", "unknown"),
                            current=s.get("current"),
                            suggested=s.get("suggested"),
                            reason=s.get("reason", ""),
                        )
                        for s in raw
                        if isinstance(s, dict)
                    ]
                    return SuggestConfigResponse(
                        suggestions=suggestions,
                        model=model_label,
                        latency_ms=latency_ms,
                        tokens_used=tokens_used,
                    )
            except Exception as exc:
                logger.warning(
                    "Azure config suggestion failed, falling back to heuristic: %s",
                    exc,
                )

        raw_suggestions = _heuristic_suggestions(
            body.current_config, body.last_metrics, body.hardware
        )
        _cache_set(cache_k, raw_suggestions, "heuristic", None)
        latency_ms = round((time.time() - start) * 1000, 1)
        suggestions = [ConfigSuggestion(**s) for s in raw_suggestions]
        return SuggestConfigResponse(
            suggestions=suggestions,
            model="heuristic",
            latency_ms=latency_ms,
            tokens_used=None,
        )
