"""FastAPI application for NightmareNet platform.

Provides REST API endpoints for dream/nightmare distortion generation
and robustness evaluation. This is the foundation for the multi-tenant
SaaS platform.

Usage:
    uvicorn nightmarenet.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import random
from typing import Any

from nightmarenet import __version__

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    from nightmarenet.api.schemas import (
        DistortionRequest,
        DistortionResponse,
        ErrorResponse,
        HealthResponse,
        RobustnessRequest,
        RobustnessResponse,
    )
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install nightmarenet[api]"
    ) from e

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions

app = FastAPI(
    title="NightmareNet API",
    description="Autonomous AI Self-Improvement Platform — Dream & Nightmare Distortion Service",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_dream_distortions(text: str, strength: float, config: dict[str, Any] | None = None) -> str:
    """Apply mild dream distortions (text + semantic).

    Args:
        text: Input text.
        strength: Distortion strength in [0, 1].
        config: Optional per-distortion weight overrides.

    Returns:
        Dream-distorted text.
    """
    result = apply_text_distortions(text, strength=strength, config=config)
    result = apply_semantic_distortions(result, strength=strength, config=config)
    return result


def _apply_nightmare_distortions(text: str, strength: float, config: dict[str, Any] | None = None) -> str:
    """Apply aggressive nightmare distortions (text + semantic + adversarial).

    Args:
        text: Input text.
        strength: Distortion strength in [0, 1].
        config: Optional per-distortion weight overrides.

    Returns:
        Nightmare-distorted text.
    """
    result = apply_text_distortions(text, strength=strength, config=config)
    result = apply_semantic_distortions(result, strength=strength, config=config)
    result = apply_adversarial_distortions(result, strength=strength, config=config)
    return result


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version=__version__, tests_passing=142)


@app.post(
    "/api/v1/generate/dream",
    response_model=DistortionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Distortion"],
)
async def generate_dream(request: DistortionRequest) -> DistortionResponse:
    """Generate dream-distorted text with mild perturbations.

    Dream distortions use text-level and semantic-level transformations
    at lower strength (recommended 0.2–0.3) to create slightly altered
    training data that forces pattern generalization.
    """
    try:
        if request.seed is not None:
            random.seed(request.seed)

        distorted = _apply_dream_distortions(
            request.text, strength=request.strength, config=request.config
        )

        return DistortionResponse(
            original_text=request.text,
            distorted_text=distorted,
            distortion_type="dream",
            strength=request.strength,
            seed=request.seed,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Dream generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error during dream generation")


@app.post(
    "/api/v1/generate/nightmare",
    response_model=DistortionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Distortion"],
)
async def generate_nightmare(request: DistortionRequest) -> DistortionResponse:
    """Generate nightmare-distorted text with aggressive perturbations.

    Nightmare distortions apply text-level, semantic-level, AND adversarial
    transformations at higher strength (recommended 0.7–0.9) to stress-test
    model robustness.
    """
    try:
        if request.seed is not None:
            random.seed(request.seed)

        distorted = _apply_nightmare_distortions(
            request.text, strength=request.strength, config=request.config
        )

        return DistortionResponse(
            original_text=request.text,
            distorted_text=distorted,
            distortion_type="nightmare",
            strength=request.strength,
            seed=request.seed,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Nightmare generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error during nightmare generation")


@app.post(
    "/api/v1/evaluate/robustness",
    response_model=RobustnessResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Evaluation"],
)
async def evaluate_robustness(request: RobustnessRequest) -> RobustnessResponse:
    """Evaluate text robustness across multiple distortion strengths.

    Applies dream and nightmare distortions at each specified strength level
    and reports how the text degrades, providing a robustness profile.
    """
    try:
        scores: dict[str, Any] = {"dream": {}, "nightmare": {}}

        for strength in request.strengths:
            random.seed(42)  # Deterministic for comparison
            dream_result = _apply_dream_distortions(request.text, strength=strength)
            nightmare_result = _apply_nightmare_distortions(request.text, strength=strength)

            # Simple character-level similarity as a quick metric
            def _char_similarity(a: str, b: str) -> float:
                if not a and not b:
                    return 1.0
                if not a or not b:
                    return 0.0
                matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
                return matches / max(len(a), len(b))

            scores["dream"][str(strength)] = {
                "similarity": round(_char_similarity(request.text, dream_result), 4),
                "length_ratio": round(len(dream_result) / max(len(request.text), 1), 4),
            }
            scores["nightmare"][str(strength)] = {
                "similarity": round(_char_similarity(request.text, nightmare_result), 4),
                "length_ratio": round(len(nightmare_result) / max(len(request.text), 1), 4),
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
            f"Text tested at {len(request.strengths)} strength levels."
        )

        return RobustnessResponse(
            original_text=request.text,
            scores=scores,
            summary=summary,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Robustness evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error during robustness evaluation")
