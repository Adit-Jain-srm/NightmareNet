"""FastAPI endpoint for natural-language experiment search."""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]
    _FASTAPI_AVAILABLE = False

from nightmarenet_server.search.embedder import ExperimentEmbedder
from nightmarenet_server.search.index import SearchIndex
from nightmarenet_server.search.query_parser import parse_query

logger = logging.getLogger(__name__)

if _FASTAPI_AVAILABLE:

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=1000)
        top_k: int = Field(10, ge=1, le=50)
        filters: Dict[str, Any] = Field(default_factory=dict)

    class SearchResult(BaseModel):
        run_id: str
        relevance_score: float
        summary: str
        metadata: Dict[str, Any]

    class SearchResponse(BaseModel):
        results: List[SearchResult]
        filters: Dict[str, Any]
        backend: str


@lru_cache(maxsize=1)
def get_embedder() -> ExperimentEmbedder:
    return ExperimentEmbedder()


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    backend = os.environ.get("NIGHTMARENET_SEARCH_BACKEND", "faiss")
    return SearchIndex(backend=backend)


def build_search_router() -> Optional[Any]:
    if not _FASTAPI_AVAILABLE:
        return None

    router = APIRouter(prefix="/api/v1/search", tags=["search"])

    @router.post("", response_model=SearchResponse)
    async def search(body: SearchRequest) -> SearchResponse:
        parsed = parse_query(body.query)
        filters = {**parsed.filters, **body.filters}
        try:
            embedding = get_embedder().embed_query(parsed.text)
            hits = get_index().hybrid_search(embedding, filters=filters, top_k=body.top_k)
        except Exception as exc:
            logger.exception("Experiment search failed")
            raise HTTPException(status_code=500, detail="Internal server error.") from exc
        return SearchResponse(
            results=[
                SearchResult(
                    run_id=hit.run_id,
                    relevance_score=round(hit.score, 6),
                    summary=_summary(hit.metadata),
                    metadata=hit.metadata,
                )
                for hit in hits
            ],
            filters=filters,
            backend=get_index().backend,
        )

    return router


def _summary(metadata: Dict[str, Any]) -> str:
    name = metadata.get("name") or metadata.get("run_id") or "experiment"
    model = metadata.get("model") or "unknown model"
    status = metadata.get("status") or "unknown"
    metrics = metadata.get("metrics") or {}
    metric_bits = []
    for key in ("accuracy", "robustness", "robustness_score", "loss"):
        if key in metrics:
            metric_bits.append(f"{key}={metrics[key]}")
    suffix = f" ({', '.join(metric_bits)})" if metric_bits else ""
    return f"{name} on {model} is {status}{suffix}."
