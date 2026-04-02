"""FastAPI application for NightmareNet platform.

Provides REST API endpoints for dream/nightmare distortion generation
and robustness evaluation. This is the foundation for the multi-tenant
SaaS platform.

Usage:
    uvicorn nightmarenet.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import random
from typing import Any, Optional

from nightmarenet import __version__

logger = logging.getLogger(__name__)

try:
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    from nightmarenet.api.auth import APIKeyMiddleware
    from nightmarenet.api.schemas import (
        DistortionRequest,
        DistortionResponse,
        ErrorResponse,
        HealthResponse,
        RobustnessRequest,
        RobustnessResponse,
    )
    from nightmarenet.distortions.adversarial import apply_adversarial_distortions
    from nightmarenet.distortions.semantic import apply_semantic_distortions
    from nightmarenet.distortions.text import apply_text_distortions
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install nightmarenet[api]"
    ) from e

_DISTORTION_BODY = Body(...)
_ROBUSTNESS_BODY = Body(...)

app = FastAPI(
    title="NightmareNet API",
    description="Autonomous AI Self-Improvement Platform — Dream & Nightmare Distortion Service",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Rate limiting ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )


# --- Authentication middleware ---
app.add_middleware(APIKeyMiddleware)

# --- CORS ---
_cors_origins = [
    o.strip()
    for o in os.environ.get("NIGHTMARENET_CORS_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_dream_distortions(
    text: str,
    strength: float,
    config: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> str:
    """Apply mild dream distortions (text + semantic).

    Args:
        text: Input text.
        strength: Distortion strength in [0, 1].
        config: Optional nested config with 'text'/'semantic' sub-keys.
        seed: Optional seed for deterministic output.

    Returns:
        Dream-distorted text.
    """
    if seed is not None:
        random.seed(seed)
    text_config = config.get("text") if config else None
    semantic_config = config.get("semantic") if config else None
    result = apply_text_distortions(text, strength=strength, config=text_config)
    result = apply_semantic_distortions(result, strength=strength, config=semantic_config)
    return result


def _apply_nightmare_distortions(
    text: str,
    strength: float,
    config: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> str:
    """Apply aggressive nightmare distortions (text + semantic + adversarial).

    Args:
        text: Input text.
        strength: Distortion strength in [0, 1].
        config: Optional nested config with 'text'/'semantic'/'adversarial' sub-keys.
        seed: Optional seed for deterministic output.

    Returns:
        Nightmare-distorted text.
    """
    if seed is not None:
        random.seed(seed)
    text_config = config.get("text") if config else None
    semantic_config = config.get("semantic") if config else None
    adversarial_config = config.get("adversarial") if config else None
    result = apply_text_distortions(text, strength=strength, config=text_config)
    result = apply_semantic_distortions(
        result, strength=strength, config=semantic_config
    )
    result = apply_adversarial_distortions(
        result, strength=strength, config=adversarial_config
    )
    return result


def _char_similarity(a: str, b: str) -> float:
    """Compute character-level similarity between two strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
    return matches / max(len(a), len(b))


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok", version=__version__
    )


@app.post(
    "/api/v1/generate/dream",
    response_model=DistortionResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Distortion"],
)
@limiter.limit("60/minute")
async def generate_dream(
    request: Request, body: DistortionRequest = _DISTORTION_BODY
) -> DistortionResponse:
    """Generate dream-distorted text with mild perturbations.

    Dream distortions use text-level and semantic-level transformations
    at lower strength (recommended 0.2–0.3) to create slightly altered
    training data that forces pattern generalization.
    """
    try:
        distorted = _apply_dream_distortions(
            body.text,
            strength=body.strength,
            config=body.config,
            seed=body.seed,
        )

        return DistortionResponse(
            original_text=body.text,
            distorted_text=distorted,
            distortion_type="dream",
            strength=body.strength,
            seed=body.seed,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Dream generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during dream generation",
        ) from None


@app.post(
    "/api/v1/generate/nightmare",
    response_model=DistortionResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Distortion"],
)
@limiter.limit("60/minute")
async def generate_nightmare(
    request: Request, body: DistortionRequest = _DISTORTION_BODY
) -> DistortionResponse:
    """Generate nightmare-distorted text with aggressive perturbations.

    Nightmare distortions apply text-level, semantic-level, AND adversarial
    transformations at higher strength (recommended 0.7–0.9) to stress-test
    model robustness.
    """
    try:
        distorted = _apply_nightmare_distortions(
            body.text,
            strength=body.strength,
            config=body.config,
            seed=body.seed,
        )

        return DistortionResponse(
            original_text=body.text,
            distorted_text=distorted,
            distortion_type="nightmare",
            strength=body.strength,
            seed=body.seed,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Nightmare generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during nightmare generation",
        ) from None


@app.post(
    "/api/v1/evaluate/robustness",
    response_model=RobustnessResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Evaluation"],
)
@limiter.limit("10/minute")
async def evaluate_robustness(
    request: Request, body: RobustnessRequest = _ROBUSTNESS_BODY
) -> RobustnessResponse:
    """Evaluate text robustness across multiple distortion strengths.

    Applies dream and nightmare distortions at each specified strength level
    and reports how the text degrades, providing a robustness profile.
    """
    try:
        scores: dict[str, Any] = {"dream": {}, "nightmare": {}}

        for i, strength in enumerate(body.strengths):
            # Use per-strength deterministic seed for reproducibility
            strength_seed = 42 + i
            dream_result = _apply_dream_distortions(
                body.text, strength=strength, seed=strength_seed
            )
            nightmare_result = _apply_nightmare_distortions(
                body.text, strength=strength, seed=strength_seed
            )

            scores["dream"][str(strength)] = {
                "similarity": round(
                    _char_similarity(body.text, dream_result), 4
                ),
                "length_ratio": round(
                    len(dream_result) / max(len(body.text), 1), 4
                ),
            }
            scores["nightmare"][str(strength)] = {
                "similarity": round(
                    _char_similarity(body.text, nightmare_result), 4
                ),
                "length_ratio": round(
                    len(nightmare_result) / max(len(body.text), 1), 4
                ),
            }

        # Summary
        avg_dream_sim = sum(
            v["similarity"] for v in scores["dream"].values()
        ) / max(len(scores["dream"]), 1)
        avg_nightmare_sim = sum(
            v["similarity"] for v in scores["nightmare"].values()
        ) / max(len(scores["nightmare"]), 1)

        summary = (
            f"Dream avg similarity: {avg_dream_sim:.2%}, "
            f"Nightmare avg similarity: {avg_nightmare_sim:.2%}. "
            f"Text tested at {len(body.strengths)} strength levels."
        )

        return RobustnessResponse(
            original_text=body.text,
            scores=scores,
            summary=summary,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Robustness evaluation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during robustness evaluation",
        ) from None
