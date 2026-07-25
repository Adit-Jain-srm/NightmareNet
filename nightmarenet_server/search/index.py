"""Persistent vector index for experiment search."""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nightmarenet_server.search.embedder import EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    run_id: str
    score: float
    metadata: Dict[str, Any]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class SearchIndex:
    """FAISS-backed index with a NumPy fallback and disk persistence."""

    def __init__(
        self,
        backend: str = "faiss",
        path: Optional[str] = None,
        dimension: int = EMBEDDING_DIM,
    ) -> None:
        self.backend = backend
        self.dimension = dimension
        self.path = Path(
            path or os.environ.get("NIGHTMARENET_SEARCH_INDEX", ".nightmarenet_search")
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._faiss: Optional[Any] = None
        self._faiss_index: Optional[Any] = None
        self._faiss_ids: List[str] = []
        self._lock = threading.RLock()
        self._load()

    def add(self, run_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        vector = _normalize(np.asarray(embedding, dtype=np.float32))
        if vector.shape != (self.dimension,):
            raise ValueError(f"expected embedding shape {(self.dimension,)}, got {vector.shape}")
        with self._lock:
            self._records[run_id] = {"run_id": run_id, "metadata": metadata}
            self._vectors[run_id] = vector
            self._rebuild_faiss()
            self.persist()

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._records.pop(run_id, None)
            self._vectors.pop(run_id, None)
            self._rebuild_faiss()
            self.persist()

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchHit]:
        with self._lock:
            if not self._vectors:
                return []
            query = _normalize(np.asarray(query_embedding, dtype=np.float32))
            if self._faiss_index is not None:
                scores, idxs = self._faiss_index.search(
                    query.reshape(1, -1),
                    min(top_k, len(self._faiss_ids)),
                )
                hits = []
                for score, idx in zip(scores[0], idxs[0]):
                    if idx < 0:
                        continue
                    run_id = self._faiss_ids[int(idx)]
                    hits.append(SearchHit(run_id, float(score), self._records[run_id]["metadata"]))
                return hits
            return self._rank_candidates(query, list(self._vectors.items()), top_k)

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]:
        filters = filters or {}
        with self._lock:
            query = _normalize(np.asarray(query_embedding, dtype=np.float32))
            candidates = [
                (run_id, vector)
                for run_id, vector in self._vectors.items()
                if _matches_filters(self._records[run_id]["metadata"], filters)
            ]
            return self._rank_candidates(query, candidates, top_k)

    def persist(self) -> None:
        with self._lock:
            payload = {
                "dimension": self.dimension,
                "records": self._records,
                "vectors": {run_id: vector.tolist() for run_id, vector in self._vectors.items()},
            }
            target = self.path / "index.json"
            fd, tmp_name = tempfile.mkstemp(
                prefix="index.",
                suffix=".tmp",
                dir=str(self.path),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    tmp.write(json.dumps(payload, default=str))
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_name, target)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def _load(self) -> None:
        with self._lock:
            index_path = self.path / "index.json"
            if not index_path.exists():
                self._rebuild_faiss()
                return
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                logger.exception("Search index could not be loaded; starting with an empty index")
                self._records = {}
                self._vectors = {}
                self._rebuild_faiss()
                return
            self.dimension = int(payload.get("dimension", self.dimension))
            self._records = dict(payload.get("records", {}))
            self._vectors = {
                run_id: _normalize(np.asarray(vector, dtype=np.float32))
                for run_id, vector in payload.get("vectors", {}).items()
            }
            self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        with self._lock:
            self._faiss_index = None
            self._faiss_ids = list(self._vectors.keys())
            if self.backend != "faiss" or not self._faiss_ids:
                return
            try:
                import faiss
            except ImportError:
                self._faiss = None
                return
            self._faiss = faiss
            index = faiss.IndexFlatIP(self.dimension)
            matrix = np.vstack([self._vectors[run_id] for run_id in self._faiss_ids]).astype(
                np.float32
            )
            index.add(matrix)
            self._faiss_index = index

    def _rank_candidates(
        self,
        query: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]],
        top_k: int,
    ) -> List[SearchHit]:
        scored = [
            SearchHit(run_id, float(np.dot(query, vector)), self._records[run_id]["metadata"])
            for run_id, vector in candidates
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


def _lookup_metric(metrics: Dict[str, Any], field: str) -> Optional[float]:
    parts = field.split(".")
    value: Any = metrics
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_term(value: Any, term: str) -> bool:
    return term in json.dumps(value, default=str).lower()


def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    status = filters.get("status")
    if status and str(metadata.get("status", "")).lower() != str(status).lower():
        return False
    model = filters.get("model")
    if model and str(model).lower() not in str(metadata.get("model", "")).lower():
        return False
    created_after = filters.get("created_after")
    if created_after and str(metadata.get("created_at", "")) < str(created_after):
        return False
    for term in filters.get("exclude_terms", []):
        if _contains_term(metadata, str(term).lower()):
            return False
    for metric_filter in filters.get("metrics", []):
        actual = _lookup_metric(metadata.get("metrics", {}), metric_filter["field"])
        if actual is None:
            actual = _lookup_metric(metadata.get("config", {}), metric_filter["field"])
        if actual is None or not _compare(actual, metric_filter["op"], metric_filter["value"]):
            return False
    return True


def _compare(actual: float, op: str, expected: float) -> bool:
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    return actual == expected
