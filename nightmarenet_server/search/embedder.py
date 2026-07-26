"""Experiment metadata embedding helpers.

The production path uses ``sentence-transformers/all-MiniLM-L6-v2`` when it is
installed. A deterministic hashing fallback keeps the OSS test suite and local
development usable without downloading model weights.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ExperimentDocument:
    """Structured representation of a run ready for embedding."""

    run_id: str
    experiment_id: str = ""
    name: str = ""
    model: str = ""
    status: str = ""
    phase: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    audit_logs: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model": self.model,
            "status": self.status,
            "phase": self.phase,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics,
            "config": self.config,
        }


def _json_loads(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _flatten(prefix: str, value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(prefix, item)
    else:
        yield f"{prefix}: {value}"


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class ExperimentEmbedder:
    """Create 384-dimensional embeddings for experiment runs and queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = EMBEDDING_DIM,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = None if model is False else model
        self._attempted_model_load = model is not None

    def embed_run(self, run: ExperimentDocument) -> np.ndarray:
        return self.embed_text(self.serialize_run(run))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)

    def embed_text(self, text: str) -> np.ndarray:
        model = self._load_model()
        if model is not None:
            vector = model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vector, dtype=np.float32)
        return self._hash_embedding(text)

    def serialize_run(self, run: ExperimentDocument) -> str:
        sections = [
            f"run_id: {run.run_id}",
            f"experiment_id: {run.experiment_id}",
            f"name: {run.name}",
            f"model: {run.model}",
            f"status: {run.status}",
            f"phase: {run.phase}",
            f"created_at: {run.created_at}",
            f"started_at: {run.started_at}",
            f"completed_at: {run.completed_at}",
            "config:",
            *list(_flatten("", run.config)),
            "metrics:",
            *list(_flatten("", run.metrics)),
            "events:",
            *[
                f"{event.get('event_type', event.get('type', 'event'))}: "
                f"{_stable_json(event.get('payload', event))}"
                for event in run.events
            ],
            "audit:",
            *[
                f"{log.get('action', 'audit')} {log.get('resource_type', '')} "
                f"{log.get('resource_id', '')}: {_stable_json(log.get('metadata', log))}"
                for log in run.audit_logs
            ],
        ]
        return "\n".join(str(s) for s in sections if s is not None)

    def _load_model(self) -> Optional[Any]:
        if self._attempted_model_load:
            return self._model
        self._attempted_model_load = True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._model = None
            return None
        self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _hash_embedding(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = [t for t in text.lower().replace("_", " ").replace("-", " ").split() if t]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + math.log1p(len(token))
            vec[idx] += sign * weight
        if not tokens:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dimension] = 1.0
        return _normalize(vec).astype(np.float32)


def document_from_orm(
    run: Any,
    experiment: Optional[Any] = None,
    audit_logs: Optional[List[Any]] = None,
) -> ExperimentDocument:
    """Build an :class:`ExperimentDocument` from SQLAlchemy rows."""
    experiment = experiment or getattr(run, "experiment", None)
    events = []
    for event in getattr(run, "events", []) or []:
        events.append(
            {
                "event_type": getattr(event, "event_type", ""),
                "payload": _json_loads(getattr(event, "payload_json", "{}")),
                "timestamp": str(getattr(event, "timestamp", "")),
            }
        )
    logs = []
    for log in audit_logs or []:
        logs.append(
            {
                "action": getattr(log, "action", ""),
                "resource_type": getattr(log, "resource_type", ""),
                "resource_id": getattr(log, "resource_id", ""),
                "metadata": _json_loads(getattr(log, "metadata_json", "{}")),
                "timestamp": str(getattr(log, "timestamp", "")),
            }
        )
    config = _json_loads(getattr(experiment, "config_json", "{}")) if experiment else {}
    metrics = _json_loads(getattr(run, "metrics_json", "{}"))
    return ExperimentDocument(
        run_id=getattr(run, "id", ""),
        experiment_id=getattr(run, "experiment_id", ""),
        name=getattr(experiment, "name", "") if experiment else "",
        model=str(
            config.get("model_name") or config.get("model") or config.get("model_type") or ""
        ),
        status=getattr(run, "status", ""),
        phase=getattr(run, "phase", ""),
        config=config,
        metrics=metrics,
        events=events,
        audit_logs=logs,
        started_at=str(getattr(run, "started_at", "") or ""),
        completed_at=str(getattr(run, "completed_at", "") or ""),
        created_at=str(getattr(experiment, "created_at", "") or ""),
    )
