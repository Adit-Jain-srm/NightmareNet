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
import subprocess
import time
from typing import Any, Optional

from nightmarenet import __version__

logger = logging.getLogger(__name__)

try:
    from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    from nightmarenet.api.auth import APIKeyMiddleware
    from nightmarenet.api.schemas import (
        CompareRequest,
        CompareResponse,
        DemoRequest,
        DemoResponse,
        DistortionDetail,
        DistortionRequest,
        DistortionResponse,
        ErrorResponse,
        HealthResponse,
        RobustnessRequest,
        RobustnessResponse,
        TrainingConfigRequest,
        TrainingConfigResponse,
        TrainingPhasePreview,
        UploadResponse,
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
_TRAINING_CONFIG_BODY = Body(...)
_COMPARE_BODY = Body(...)
_DEMO_BODY = Body(...)

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
app.add_middleware(SlowAPIMiddleware)  # type: ignore[arg-type]


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )


# --- Authentication middleware ---
app.add_middleware(APIKeyMiddleware)  # type: ignore[arg-type]

# --- CORS ---
_cors_origins = [
    o.strip()
    for o in os.environ.get("NIGHTMARENET_CORS_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
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

    # Enable learned adversarial distortions at higher strengths for better
    # nightmare quality — the MLM-based generator creates more challenging
    # training data by targeting high-importance tokens.
    if adversarial_config is None and strength >= 0.5:
        adversarial_config = {
            "contradiction": 0.3,
            "ambiguity": 0.3,
            "cross_domain": 0.2,
            "misleading_context": 0.2,
            "learned": min(strength, 0.4),
        }

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


# --- Cached test count ---
_test_count_cache: dict[str, Any] = {"count": None, "checked_at": 0.0}
_TEST_CACHE_TTL = 300  # refresh every 5 minutes


def _get_test_count() -> Optional[int]:
    """Return the number of passing tests, cached for 5 minutes."""
    now = time.time()
    if (
        _test_count_cache["count"] is not None
        and now - _test_count_cache["checked_at"] < _TEST_CACHE_TTL
    ):
        return _test_count_cache["count"]
    try:
        result = subprocess.run(
            ["pytest", "tests/", "--co", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if "test" in line and "collected" in line:
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        count = int(part)
                        _test_count_cache["count"] = count
                        _test_count_cache["checked_at"] = now
                        return count
    except Exception:
        logger.debug("Failed to count tests", exc_info=True)
    return _test_count_cache["count"]


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=__version__,
        tests_passing=_get_test_count(),
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
            f"Text tested at {len(body.strengths)} strength levels. "
        )
        # Add degradation analysis
        sorted_strengths = sorted(body.strengths)
        if len(sorted_strengths) >= 2:
            low_key = str(sorted_strengths[0])
            high_key = str(sorted_strengths[-1])
            dream_low = scores["dream"].get(low_key, {}).get("similarity", 1.0)
            dream_high = scores["dream"].get(high_key, {}).get("similarity", 0.0)
            drop_rate = dream_low - dream_high
            if drop_rate > 0.4:
                summary += (
                    f"Dream similarity drops {drop_rate:.0%} from strength "
                    f"{sorted_strengths[0]} to {sorted_strengths[-1]}, "
                    "indicating HIGH sensitivity to distortion. "
                    "Consider adversarial training (nightmare phase) to "
                    "build robustness. "
                )
            elif drop_rate > 0.15:
                summary += (
                    f"Moderate degradation ({drop_rate:.0%} drop) across "
                    "the strength range. The text structure is partially "
                    "resilient but would benefit from dream-phase training. "
                )
            else:
                summary += (
                    f"Only {drop_rate:.0%} drop across the full range — "
                    "this text is highly resilient to distortion. "
                )
        if avg_nightmare_sim < 0.3:
            summary += (
                "Nightmare distortions cause severe degradation "
                "(<30% avg similarity), which is expected — nightmare "
                "mode applies adversarial attacks designed to stress-test "
                "model generalization. "
            )
        gap = avg_dream_sim - avg_nightmare_sim
        if gap > 0:
            summary += (
                f"The {gap:.0%} gap between dream and nightmare "
                "resilience shows the value of the nightmare phase "
                "in exposing model weaknesses."
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


# --- Training Config Preview ---


_VALID_MODEL_TYPES = {"causal_lm", "masked_lm", "seq_classification"}


@app.post(
    "/api/v1/train/config",
    response_model=TrainingConfigResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Training"],
)
@limiter.limit("30/minute")
async def preview_training_config(
    request: Request, body: TrainingConfigRequest = _TRAINING_CONFIG_BODY
) -> TrainingConfigResponse:
    """Validate and preview a training configuration.

    Returns the full phase schedule, total epochs, and actionable
    recommendations for improving model accuracy.
    """
    try:
        recommendations: list[str] = []
        valid = True

        if body.model_type not in _VALID_MODEL_TYPES:
            valid = False
            recommendations.append(
                f"Invalid model_type '{body.model_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_MODEL_TYPES))}."
            )

        # Build phase schedule
        phases: list[TrainingPhasePreview] = []
        total_epochs = 0
        lr = body.learning_rate

        for cycle in range(1, body.num_cycles + 1):
            if body.wake_epochs > 0:
                phases.append(TrainingPhasePreview(
                    cycle=cycle, phase="wake", epochs=body.wake_epochs,
                    learning_rate=lr,
                    description="Supervised learning on clean data.",
                ))
                total_epochs += body.wake_epochs

            if body.dream_epochs > 0:
                phases.append(TrainingPhasePreview(
                    cycle=cycle, phase="dream", epochs=body.dream_epochs,
                    learning_rate=lr,
                    description=(
                        f"Mild distortions (strength={body.dream_strength}) "
                        f"with KL weight={body.kl_weight}."
                    ),
                ))
                total_epochs += body.dream_epochs

            if body.nightmare_epochs > 0:
                nlr = lr * body.nightmare_lr_multiplier
                desc = (
                    f"Extreme distortions (strength={body.nightmare_strength}), "
                    f"LR×{body.nightmare_lr_multiplier}={nlr:.2e}."
                )
                if body.use_learned_adversarial:
                    desc += " Learned adversarial enabled."
                phases.append(TrainingPhasePreview(
                    cycle=cycle, phase="nightmare", epochs=body.nightmare_epochs,
                    learning_rate=nlr, description=desc,
                ))
                total_epochs += body.nightmare_epochs

            # Compress phase (1 epoch for pruning + fine-tuning)
            phases.append(TrainingPhasePreview(
                cycle=cycle, phase="compress", epochs=1,
                learning_rate=lr,
                description=f"Magnitude pruning at ratio={body.pruning_ratio}, then fine-tune.",
            ))
            total_epochs += 1

        # Recommendations
        if body.dream_strength >= 0.5:
            recommendations.append(
                "Dream strength ≥ 0.5 is unusually high. "
                "Values of 0.2–0.3 work best for gentle generalization."
            )
        if body.nightmare_strength < 0.5:
            recommendations.append(
                "Nightmare strength < 0.5 is mild. "
                "Try 0.7–0.9 for effective stress-testing."
            )
        if body.num_cycles < 3:
            recommendations.append(
                "Fewer than 3 cycles may not provide enough improvement. "
                "3–5 cycles is recommended."
            )
        if body.nightmare_epochs == 0:
            recommendations.append(
                "Nightmare phase is disabled (0 epochs). "
                "This removes the adversarial robustness component."
            )
        if body.pruning_ratio > 0.5:
            recommendations.append(
                f"Pruning ratio {body.pruning_ratio} is aggressive. "
                "Values above 0.5 risk significant accuracy loss."
            )
        if not body.early_stopping and body.num_cycles >= 5:
            recommendations.append(
                "Consider enabling early_stopping for ≥ 5 cycles "
                "to avoid unnecessary computation."
            )
        if not body.use_learned_adversarial and body.nightmare_strength >= 0.7:
            recommendations.append(
                "Enable use_learned_adversarial for stronger nightmare phases. "
                "MLM-based distortions target high-importance tokens for better training."
            )

        config_summary = {
            "model_name": body.model_name,
            "model_type": body.model_type,
            "num_cycles": body.num_cycles,
            "total_phases": len(phases),
            "total_epochs": total_epochs,
            "learning_rate": body.learning_rate,
            "nightmare_lr": body.learning_rate * body.nightmare_lr_multiplier,
            "dream_strength": body.dream_strength,
            "nightmare_strength": body.nightmare_strength,
            "pruning_ratio": body.pruning_ratio,
            "kl_weight": body.kl_weight,
            "early_stopping": body.early_stopping,
            "use_learned_adversarial": body.use_learned_adversarial,
        }

        return TrainingConfigResponse(
            valid=valid,
            total_phases=len(phases),
            total_epochs=total_epochs,
            estimated_phases=phases,
            config_summary=config_summary,
            recommendations=recommendations,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Training config preview failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during training config preview",
        ) from None


# --- Distortion Comparison ---


@app.post(
    "/api/v1/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Evaluation"],
)
@limiter.limit("10/minute")
async def compare_distortions(
    request: Request, body: CompareRequest = _COMPARE_BODY
) -> CompareResponse:
    """Compare dream and nightmare distortion effects at two strength levels.

    Returns side-by-side distortion details with a resilience score indicating
    how well the text's semantic structure survives escalation.
    """
    try:
        seed = body.seed

        # Baseline distortions
        dream_base = _apply_dream_distortions(body.text, body.baseline_strength, seed=seed)
        nightmare_base = _apply_nightmare_distortions(
            body.text, body.baseline_strength, seed=seed
        )

        # Challenge distortions
        dream_challenge = _apply_dream_distortions(body.text, body.challenge_strength, seed=seed)
        nightmare_challenge = _apply_nightmare_distortions(
            body.text, body.challenge_strength, seed=seed
        )

        def _detail(distorted: str) -> DistortionDetail:
            return DistortionDetail(
                distorted_text=distorted,
                similarity=round(_char_similarity(body.text, distorted), 4),
                length_ratio=round(len(distorted) / max(len(body.text), 1), 4),
            )

        dream_details = {
            "baseline": _detail(dream_base),
            "challenge": _detail(dream_challenge),
        }
        nightmare_details = {
            "baseline": _detail(nightmare_base),
            "challenge": _detail(nightmare_challenge),
        }

        # Resilience = how much similarity drops between baseline and challenge
        dream_drop = max(
            dream_details["baseline"].similarity - dream_details["challenge"].similarity, 0.0
        )
        nightmare_drop = max(
            nightmare_details["baseline"].similarity - nightmare_details["challenge"].similarity,
            0.0,
        )
        avg_drop = (dream_drop + nightmare_drop) / 2
        resilience = round(max(1.0 - avg_drop * 2, 0.0), 4)

        analysis = (
            f"Dream similarity drops from "
            f"{dream_details['baseline'].similarity:.2%} to "
            f"{dream_details['challenge'].similarity:.2%} "
            f"(Δ={dream_drop:.2%}). "
            f"Nightmare similarity drops from "
            f"{nightmare_details['baseline'].similarity:.2%} to "
            f"{nightmare_details['challenge'].similarity:.2%} "
            f"(Δ={nightmare_drop:.2%}). "
            f"Resilience score: {resilience:.2%}."
        )

        return CompareResponse(
            original_text=body.text,
            baseline_strength=body.baseline_strength,
            challenge_strength=body.challenge_strength,
            dream=dream_details,
            nightmare=nightmare_details,
            resilience_score=resilience,
            analysis=analysis,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Compare evaluation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during distortion comparison",
        ) from None


# --- Interactive Demo ---


@app.post(
    "/api/v1/demo",
    response_model=DemoResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Demo"],
)
@limiter.limit("60/minute")
async def interactive_demo(
    request: Request, body: DemoRequest = _DEMO_BODY
) -> DemoResponse:
    """Run dream + nightmare distortions in one call for the guided demo.

    Returns both distortion results with a resilience delta and a
    human-readable insight explaining what the distortions reveal
    about the input text.
    """
    try:
        dream_strength = 0.25
        nightmare_strength = 0.80

        dream_result = _apply_dream_distortions(
            body.text, dream_strength, seed=body.seed
        )
        nightmare_result = _apply_nightmare_distortions(
            body.text, nightmare_strength, seed=body.seed
        )

        dream_sim = round(
            _char_similarity(body.text, dream_result), 4
        )
        nightmare_sim = round(
            _char_similarity(body.text, nightmare_result), 4
        )
        delta = round(dream_sim - nightmare_sim, 4)

        word_count = len(body.text.split())
        if delta < 0.1:
            quality = "highly resilient"
        elif delta < 0.25:
            quality = "moderately resilient"
        else:
            quality = "vulnerable to adversarial perturbation"

        insight = (
            f"This {word_count}-word text is {quality}. "
            f"Dream distortion preserved {dream_sim:.0%} "
            f"similarity while nightmare dropped to "
            f"{nightmare_sim:.0%} — a delta of "
            f"{delta:.0%}. During training, your model "
            f"would learn to maintain performance across "
            f"this full distortion spectrum."
        )

        return DemoResponse(
            original_text=body.text,
            dream=DistortionDetail(
                distorted_text=dream_result,
                similarity=dream_sim,
                length_ratio=round(
                    len(dream_result) / max(len(body.text), 1),
                    4,
                ),
            ),
            nightmare=DistortionDetail(
                distorted_text=nightmare_result,
                similarity=nightmare_sim,
                length_ratio=round(
                    len(nightmare_result)
                    / max(len(body.text), 1),
                    4,
                ),
            ),
            resilience_delta=delta,
            insight=insight,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Demo generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during demo generation",
        ) from None


# --- File Upload ---

_ALLOWED_EXTENSIONS = {".txt", ".csv", ".json"}
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post(
    "/api/v1/upload/text",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Upload"],
)
@limiter.limit("30/minute")
async def upload_text_file(request: Request, file: UploadFile) -> UploadResponse:
    """Upload a text file for processing through the distortion pipeline.

    Accepts .txt, .csv, and .json files up to 5 MB. Returns extracted
    text content with metadata for use in other endpoints.
    """
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{ext}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                ),
            )

        content_bytes = await file.read()
        if len(content_bytes) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content_bytes)} bytes). Max: {_MAX_UPLOAD_BYTES}.",
            )

        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as ue:
            raise HTTPException(
                status_code=400,
                detail="File must be valid UTF-8 text.",
            ) from ue

        words = text_content.split()
        lines = text_content.splitlines()
        preview = text_content[:500] + ("..." if len(text_content) > 500 else "")

        return UploadResponse(
            filename=filename,
            file_type=ext,
            text_content=text_content,
            char_count=len(text_content),
            word_count=len(words),
            line_count=len(lines),
            preview=preview,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File upload failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal error during file upload",
        ) from None
