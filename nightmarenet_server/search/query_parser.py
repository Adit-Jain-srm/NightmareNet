"""Natural-language search query parsing."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


@dataclass
class ParsedQuery:
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    terms: List[str] = field(default_factory=list)


_STATUS_RE = re.compile(r"\b(completed|complete|failed|running|queued|pending)\b", re.I)
_MODEL_RE = re.compile(r"\b(?:model|using|used)\s+([A-Za-z0-9_.\-/]+)", re.I)
_METRIC_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.-]*(?:\s+[A-Za-z_][A-Za-z0-9_.-]*)?)"
    r"\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)",
    re.I,
)
_FIELD_PREFIXES = {"and", "or", "where", "with", "having", "whose"}


def _normalize_field(raw: str) -> str:
    parts = raw.lower().split()
    while len(parts) > 1 and parts[0] in _FIELD_PREFIXES:
        parts = parts[1:]
    return "_".join(parts)


def parse_query(query: str) -> ParsedQuery:
    """Extract lightweight structured filters from natural language."""
    filters: Dict[str, Any] = {}
    terms: List[str] = []
    raw = query.strip()

    status_match = _STATUS_RE.search(raw)
    if status_match:
        status = status_match.group(1).lower()
        filters["status"] = "completed" if status == "complete" else status

    model_match = _MODEL_RE.search(raw)
    if model_match:
        filters["model"] = model_match.group(1).lower()

    metrics = []
    for metric, op, raw_value in _METRIC_RE.findall(raw):
        metrics.append(
            {
                "field": _normalize_field(metric),
                "op": op,
                "value": float(raw_value),
            }
        )
    if metrics:
        filters["metrics"] = metrics

    if "last week" in raw.lower():
        filters["created_after"] = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    not_terms = re.findall(r"\bnot\s+([A-Za-z0-9_.-]+)", raw, flags=re.I)
    if not_terms:
        filters["exclude_terms"] = [t.lower() for t in not_terms]

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", raw):
        terms.append(token.lower())

    return ParsedQuery(text=raw, filters=filters, terms=terms)
