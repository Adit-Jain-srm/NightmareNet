"""Data optimization endpoint — Adaption Labs integration.

Provides endpoints to optimize text datasets via the Adaption Labs SDK,
with SSE progress streaming, async execution, status tracking, and
proper error handling for all failure modes.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Data Optimization"])

_OPTIMIZE_BODY = Body(...)
_STREAM_BODY = Body(...)
_IMPORT_BODY = Body(...)
_ESTIMATE_BODY = Body(...)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="data-opt")

_MAX_REGISTRY_SIZE = int(os.environ.get("NIGHTMARENET_MAX_OPTIMIZE_RUNNERS", "32"))


class OptimizationState(str, Enum):
    PENDING = "pending"
    ESTIMATING = "estimating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationRun:
    """Tracks a single data optimization run."""

    def __init__(self, run_id: str, text_count: int) -> None:
        self.run_id = run_id
        self.state = OptimizationState.PENDING
        self.text_count = text_count
        self.progress_pct: float = 0.0
        self.message: str = "Queued"
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.before_stats: Optional[Dict[str, Any]] = None
        self.after_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        elapsed = None
        if self.started_at:
            end = self.completed_at or time.time()
            elapsed = round(end - self.started_at, 2)
        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "text_count": self.text_count,
            "progress_pct": round(self.progress_pct, 1),
            "message": self.message,
            "elapsed_seconds": elapsed,
            "error": self.error,
            "before_stats": self.before_stats,
            "after_stats": self.after_stats,
        }


_registry: Dict[str, OptimizationRun] = {}


def _evict_completed() -> None:
    """Remove oldest completed runs when registry is at capacity."""
    if len(_registry) < _MAX_REGISTRY_SIZE:
        return
    terminal_states = (
        OptimizationState.COMPLETED,
        OptimizationState.FAILED,
        OptimizationState.CANCELLED,
    )
    completed = [(rid, r) for rid, r in _registry.items() if r.state in terminal_states]
    completed.sort(key=lambda x: x[1].completed_at or 0)
    while len(_registry) >= _MAX_REGISTRY_SIZE and completed:
        rid, _ = completed.pop(0)
        del _registry[rid]


class DataOptimizeRequest(BaseModel):
    """Request model for dataset optimization."""

    texts: List[str] = Field(..., min_length=1, max_length=10000)
    column_mapping: Dict[str, Any]
    phase: Optional[str] = Field(None, pattern="^(wake|dream|nightmare|compress)$")
    brand_controls: Optional[Dict[str, Any]] = None
    recipe_specification: Optional[Dict[str, Any]] = None
    job_specification: Optional[Dict[str, Any]] = None
    estimate_only: bool = False

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least 1 text is required")
        if len(v) > 10000:
            raise ValueError("Maximum 10,000 texts allowed")
        return v

    @field_validator("column_mapping")
    @classmethod
    def _must_have_prompt_key(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" not in v:
            raise ValueError("column_mapping must include a 'prompt' key")
        return v


class DataImportRequest(BaseModel):
    """Request for HuggingFace/Kaggle import optimization."""

    source: str = Field(..., pattern="^(huggingface|kaggle)$")
    url: str
    files: List[str] = Field(..., min_length=1)
    column_mapping: Dict[str, Any]
    brand_controls: Optional[Dict[str, Any]] = None
    recipe_specification: Optional[Dict[str, Any]] = None


class DataOptimizeResponse(BaseModel):
    """Response model for optimization results."""

    status: str
    run_id: Optional[str] = None
    optimized_count: Optional[int] = None
    quality: Optional[Dict[str, Any]] = None
    estimate: Optional[Dict[str, Any]] = None
    elapsed_seconds: Optional[float] = None
    before_stats: Optional[Dict[str, Any]] = None
    after_stats: Optional[Dict[str, Any]] = None
    quality_delta: Optional[Dict[str, Any]] = None


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization run status."""

    run_id: str
    state: str
    text_count: int
    progress_pct: float
    message: str
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None
    before_stats: Optional[Dict[str, Any]] = None
    after_stats: Optional[Dict[str, Any]] = None


def _compute_text_stats(texts: List[str]) -> Dict[str, Any]:
    """Compute basic statistics about a text corpus."""
    count = len(texts)
    if count == 0:
        return {"count": 0, "avg_length": 0, "total_chars": 0, "avg_words": 0}
    lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    return {
        "count": count,
        "avg_length": round(sum(lengths) / count, 1),
        "total_chars": sum(lengths),
        "avg_words": round(sum(word_counts) / count, 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def _run_optimization_sync(
    texts: List[str],
    column_mapping: Dict[str, Any],
    run: OptimizationRun,
    brand_controls: Optional[Dict[str, Any]] = None,
    recipe_specification: Optional[Dict[str, Any]] = None,
    job_specification: Optional[Dict[str, Any]] = None,
) -> None:
    """Execute optimization synchronously (called in thread pool)."""
    from nightmarenet.data.adaption import Adaption, AdaptionOptimizer

    run.state = OptimizationState.RUNNING
    run.started_at = time.time()
    run.message = "Initializing optimizer"
    run.progress_pct = 5.0
    run.before_stats = _compute_text_stats(texts)

    if Adaption is None:
        run.state = OptimizationState.FAILED
        run.error = "Adaption SDK not installed. Install with: pip install adaption"
        run.completed_at = time.time()
        return

    if not os.environ.get("ADAPTION_API_KEY"):
        run.state = OptimizationState.FAILED
        run.error = "ADAPTION_API_KEY environment variable not set."
        run.completed_at = time.time()
        return

    try:
        from datasets import Dataset
    except ImportError:
        run.state = OptimizationState.FAILED
        run.error = "HuggingFace datasets library not available."
        run.completed_at = time.time()
        return

    try:
        run.message = "Building dataset"
        run.progress_pct = 15.0
        dataset = Dataset.from_dict({"text": texts})

        optimizer = AdaptionOptimizer()

        run.message = "Uploading to Adaption Labs"
        run.progress_pct = 30.0

        result = optimizer.optimize_dataset(
            dataset,
            column_mapping,
            max_rows=len(texts),
            brand_controls=brand_controls,
            recipe_specification=recipe_specification,
            job_specification=job_specification,
        )

        if result is None:
            run.state = OptimizationState.FAILED
            run.error = "Optimization returned no result. Check API key and dataset format."
            run.completed_at = time.time()
            return

        run.progress_pct = 90.0
        run.message = "Processing results"

        optimized_dataset, quality = result
        has_text_col = "text" in optimized_dataset.column_names
        optimized_texts = optimized_dataset["text"] if has_text_col else []
        run.after_stats = _compute_text_stats(list(optimized_texts)) if optimized_texts else None

        run.result = {
            "optimized_count": len(optimized_dataset),
            "quality": quality,
        }
        run.state = OptimizationState.COMPLETED
        run.progress_pct = 100.0
        run.message = "Optimization complete"
        run.completed_at = time.time()

    except TimeoutError:
        run.state = OptimizationState.FAILED
        run.error = "Optimization timed out after 10 minutes."
        run.completed_at = time.time()
    except Exception as exc:
        logger.warning("Data optimization failed: %s", exc, exc_info=True)
        run.state = OptimizationState.FAILED
        run.error = f"Optimization error: {str(exc)}"
        run.completed_at = time.time()


def register_data_optimize_routes(app: Any, limiter: Limiter) -> None:
    """Mount data optimization routes on the FastAPI app."""

    @router.post(
        "/api/v1/data/optimize",
        response_model=DataOptimizeResponse,
        responses={
            400: {"description": "Bad request — invalid input"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "Adaption SDK or dependency unavailable"},
            504: {"description": "Optimization timed out"},
            500: {"description": "Internal server error"},
        },
    )
    @limiter.limit("5/minute")
    async def optimize_data(
        request: Request,
        body: DataOptimizeRequest = _OPTIMIZE_BODY,
    ) -> DataOptimizeResponse:
        """Optimize a text dataset via Adaption Labs.

        When ``estimate_only=True``, returns credit cost without running
        the full optimization. Otherwise runs the optimization in a
        background thread and returns results with timing and quality deltas.
        """
        from nightmarenet.data.adaption import Adaption, AdaptionOptimizer

        if Adaption is None:
            raise HTTPException(
                status_code=503,
                detail="Adaption SDK not installed. Install with: pip install adaption",
            )
        if not os.environ.get("ADAPTION_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="ADAPTION_API_KEY environment variable not set.",
            )

        try:
            from datasets import Dataset
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="HuggingFace datasets library not available.",
            ) from exc

        start_time = time.time()
        before_stats = _compute_text_stats(body.texts)

        dataset = Dataset.from_dict({"text": body.texts})
        optimizer = AdaptionOptimizer()

        if body.estimate_only:
            try:
                estimate = await asyncio.get_running_loop().run_in_executor(
                    _executor, lambda: optimizer.estimate_cost(dataset, body.column_mapping)
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Cost estimation failed: {str(exc)}",
                ) from exc

            if estimate is None:
                raise HTTPException(
                    status_code=503,
                    detail="Cost estimation failed. Check API key and SDK.",
                )
            elapsed = round(time.time() - start_time, 2)
            return DataOptimizeResponse(
                status="estimated",
                estimate=estimate,
                elapsed_seconds=elapsed,
                before_stats=before_stats,
            )

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    _executor,
                    lambda: optimizer.optimize_dataset(
                        dataset,
                        body.column_mapping,
                        max_rows=len(body.texts),
                        brand_controls=body.brand_controls,
                        recipe_specification=body.recipe_specification,
                        job_specification=body.job_specification,
                    ),
                ),
                timeout=660.0,
            )
        except asyncio.TimeoutError as te:
            raise HTTPException(
                status_code=504,
                detail="Optimization timed out after 10 minutes.",
            ) from te
        except Exception as exc:
            logger.exception("Data optimization error: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Internal optimization error: {str(exc)}",
            ) from exc

        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Optimization failed. Check API key, SDK, and dataset format.",
            )

        optimized_dataset, quality = result
        elapsed = round(time.time() - start_time, 2)

        optimized_texts = (
            list(optimized_dataset["text"]) if "text" in optimized_dataset.column_names else []
        )
        after_stats = _compute_text_stats(optimized_texts) if optimized_texts else None

        quality_delta = None
        if after_stats and before_stats:
            quality_delta = {
                "count_change": after_stats["count"] - before_stats["count"],
                "avg_length_change": round(
                    after_stats["avg_length"] - before_stats["avg_length"], 1
                ),
            }

        run_id = str(uuid.uuid4())[:8]
        return DataOptimizeResponse(
            status="completed",
            run_id=run_id,
            optimized_count=len(optimized_dataset),
            quality=quality,
            elapsed_seconds=elapsed,
            before_stats=before_stats,
            after_stats=after_stats,
            quality_delta=quality_delta,
        )

    @router.post(
        "/api/v1/data/optimize/start",
        response_model=OptimizationStatusResponse,
        responses={
            400: {"description": "Bad request"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "Registry at capacity"},
        },
    )
    @limiter.limit("5/minute")
    async def start_optimization(
        request: Request,
        body: DataOptimizeRequest = _OPTIMIZE_BODY,
    ) -> OptimizationStatusResponse:
        """Start an async optimization run and return a run_id for polling."""
        _evict_completed()
        if len(_registry) >= _MAX_REGISTRY_SIZE:
            raise HTTPException(
                status_code=503,
                detail="Optimization registry at capacity. Try again later.",
            )

        run_id = str(uuid.uuid4())[:12]
        run = OptimizationRun(run_id=run_id, text_count=len(body.texts))
        _registry[run_id] = run

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            _executor,
            _run_optimization_sync,
            body.texts,
            body.column_mapping,
            run,
            body.brand_controls,
            body.recipe_specification,
            body.job_specification,
        )

        return OptimizationStatusResponse(**run.to_dict())

    @router.get(
        "/api/v1/data/optimize/{run_id}/status",
        response_model=OptimizationStatusResponse,
        responses={
            404: {"description": "Run not found"},
        },
    )
    async def get_optimization_status(run_id: str) -> OptimizationStatusResponse:
        """Poll the status of an async optimization run."""
        run = _registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Optimization run '{run_id}' not found")
        return OptimizationStatusResponse(**run.to_dict())

    @router.post(
        "/api/v1/data/optimize/stream",
        responses={
            400: {"description": "Bad request"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "SDK unavailable"},
        },
    )
    @limiter.limit("5/minute")
    async def optimize_data_stream(
        request: Request,
        body: DataOptimizeRequest = _STREAM_BODY,
    ) -> StreamingResponse:
        """Stream optimization progress as Server-Sent Events.

        Emits events with ``{state, progress_pct, message}`` until
        completion, then a final event with the full result.
        """
        run_id = str(uuid.uuid4())[:12]
        run = OptimizationRun(run_id=run_id, text_count=len(body.texts))
        _registry[run_id] = run

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            _executor,
            _run_optimization_sync,
            body.texts,
            body.column_mapping,
            run,
            body.brand_controls,
            body.recipe_specification,
            body.job_specification,
        )

        async def event_generator():
            last_pct = -1.0
            yield _sse_encode({"event": "start", "run_id": run_id})

            active_states = (
                OptimizationState.PENDING,
                OptimizationState.ESTIMATING,
                OptimizationState.RUNNING,
            )
            while run.state in active_states:
                if run.progress_pct != last_pct:
                    last_pct = run.progress_pct
                    yield _sse_encode(
                        {
                            "event": "progress",
                            "run_id": run_id,
                            "state": run.state.value,
                            "progress_pct": round(run.progress_pct, 1),
                            "message": run.message,
                        }
                    )
                await asyncio.sleep(0.5)

            if run.state == OptimizationState.COMPLETED:
                yield _sse_encode(
                    {
                        "event": "complete",
                        "run_id": run_id,
                        "state": "completed",
                        "progress_pct": 100.0,
                        "message": run.message,
                        "result": run.result,
                        "before_stats": run.before_stats,
                        "after_stats": run.after_stats,
                        "elapsed_seconds": round(
                            (run.completed_at or time.time()) - (run.started_at or time.time()), 2
                        ),
                    }
                )
            else:
                yield _sse_encode(
                    {
                        "event": "error",
                        "run_id": run_id,
                        "state": run.state.value,
                        "error": run.error or "Unknown failure",
                    }
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.post(
        "/api/v1/data/import",
        responses={
            400: {"description": "Bad request"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "SDK unavailable"},
        },
    )
    @limiter.limit("3/minute")
    async def import_and_optimize(
        request: Request,
        body: DataImportRequest = _IMPORT_BODY,
    ) -> DataOptimizeResponse:
        """Import from HuggingFace/Kaggle and optimize via Adaption."""
        from nightmarenet.data.adaption import Adaption, AdaptionOptimizer

        if Adaption is None:
            raise HTTPException(status_code=503, detail="Adaption SDK not installed.")
        if not os.environ.get("ADAPTION_API_KEY"):
            raise HTTPException(status_code=503, detail="ADAPTION_API_KEY not set.")

        start_time = time.time()
        optimizer = AdaptionOptimizer()

        try:
            if body.source == "huggingface":
                result = await asyncio.get_running_loop().run_in_executor(
                    _executor,
                    lambda: optimizer.optimize_from_huggingface(
                        body.url,
                        body.files,
                        body.column_mapping,
                        brand_controls=body.brand_controls,
                        recipe_specification=body.recipe_specification,
                    ),
                )
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    _executor,
                    lambda: optimizer.optimize_from_kaggle(
                        body.url,
                        body.files,
                        body.column_mapping,
                        brand_controls=body.brand_controls,
                        recipe_specification=body.recipe_specification,
                    ),
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Import failed: {exc}") from exc

        if result is None:
            raise HTTPException(status_code=503, detail="Import optimization failed.")

        dataset_id, quality = result
        elapsed = round(time.time() - start_time, 2)

        return DataOptimizeResponse(
            status="completed",
            run_id=dataset_id,
            quality=quality,
            elapsed_seconds=elapsed,
        )

    @router.post(
        "/api/v1/data/optimize/estimate",
        responses={
            400: {"description": "Bad request"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "SDK unavailable"},
        },
    )
    @limiter.limit("10/minute")
    async def estimate_optimization(
        request: Request,
        body: DataOptimizeRequest = _ESTIMATE_BODY,
    ) -> DataOptimizeResponse:
        """Estimate optimization cost without starting a run."""
        from nightmarenet.data.adaption import Adaption, AdaptionOptimizer

        if Adaption is None:
            raise HTTPException(status_code=503, detail="Adaption SDK not installed.")
        if not os.environ.get("ADAPTION_API_KEY"):
            raise HTTPException(status_code=503, detail="ADAPTION_API_KEY not set.")

        try:
            from datasets import Dataset
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="datasets library not available.") from exc

        dataset = Dataset.from_dict({"text": body.texts})
        optimizer = AdaptionOptimizer()

        start_time = time.time()
        try:
            estimate = await asyncio.get_running_loop().run_in_executor(
                _executor,
                lambda: optimizer.estimate_cost(
                    dataset,
                    body.column_mapping,
                    brand_controls=body.brand_controls,
                    recipe_specification=body.recipe_specification,
                    job_specification=body.job_specification,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Estimation failed: {exc}") from exc

        if estimate is None:
            raise HTTPException(status_code=503, detail="Estimation failed.")

        elapsed = round(time.time() - start_time, 2)
        return DataOptimizeResponse(
            status="estimated",
            estimate=estimate,
            elapsed_seconds=elapsed,
            before_stats=_compute_text_stats(body.texts),
        )

    app.include_router(router)


def _sse_encode(data: Dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
