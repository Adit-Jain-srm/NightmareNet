"""Robustness badge endpoints (shields.io-style).

Renders a small SVG badge showing a robustness score in the standard
"label | value" layout that drop-cleanly into a GitHub README, plus a
JSON variant for clients that want to render their own badge.

Mounted at ``/api/v1/badge`` by :mod:`nightmarenet.api.app`. The route
intentionally has no API-key requirement (badges are embedded on public
READMEs) and ships a generous ``Cache-Control`` so CDNs can absorb the
traffic.
"""

import json
import logging
from typing import Optional, Tuple, Union
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import Response
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install nightmarenet[api]"
    ) from e


router = APIRouter(prefix="/api/v1/badge", tags=["Badge"])


# Shields.io-style colour scale, ordered from best → worst.
_BANDS: Tuple[Tuple[float, str, str], ...] = (
    (0.90, "#22c55e", "elite"),
    (0.75, "#84cc16", "strong"),
    (0.60, "#eab308", "fair"),
    (0.40, "#f97316", "weak"),
    (0.00, "#ef4444", "critical"),
)

_LABEL_TEXT = "robustness"
# Pixel widths sized for Verdana 11px. Conservative; matches shields.io
# rendering closely enough that it sits cleanly next to other badges.
_LABEL_WIDTH = 78
_VALUE_WIDTH = 46
_TOTAL_WIDTH = _LABEL_WIDTH + _VALUE_WIDTH
_HEIGHT = 20
_CACHE_HEADER = "public, max-age=300"
_SVG_CONTENT_TYPE = "image/svg+xml; charset=utf-8"


def _classify(score: float) -> Tuple[str, str]:
    """Return ``(color, label)`` for a normalised score in ``[0, 1]``."""
    for threshold, color, label in _BANDS:
        if score >= threshold:
            return color, label
    return _BANDS[-1][1], _BANDS[-1][2]


def _validate_score(score: float) -> float:
    """Raise 400 if the score is outside ``[0, 1]`` (or NaN)."""
    try:
        s = float(score)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="score must be a real number in [0, 1]") from e
    # NaN comparisons always return False; trap explicitly.
    if not (s == s):  # noqa: PLR0124
        raise HTTPException(status_code=400, detail="score must be a real number in [0, 1]")
    if s < 0.0 or s > 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"score must be in [0, 1]; got {s}",
        )
    return s


def _render_svg(score_or_text: Union[float, str], color: str) -> str:
    """Render the badge as an SVG string. Inputs are trusted (already validated)."""
    if isinstance(score_or_text, (int, float)):
        value_text = f"{score_or_text:.2f}"
    else:
        value_text = str(score_or_text)

    label_safe = escape(_LABEL_TEXT)
    value_safe = escape(value_text)
    color_safe = escape(color)

    val_width = max(_VALUE_WIDTH, len(value_text) * 7 + 10)
    total_width = _LABEL_WIDTH + val_width

    label_cx = _LABEL_WIDTH / 2
    value_cx = _LABEL_WIDTH + val_width / 2

    # Render text twice (shadow + main) for the classic shields.io
    # legibility trick — a 1px-down dark shadow improves contrast on
    # GitHub's light *and* dark themes.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_width}" height="{_HEIGHT}" '
        f'role="img" aria-label="{label_safe}: {value_safe}">'
        f"<title>{label_safe}: {value_safe}</title>"
        f'<linearGradient id="s" x2="0" y2="100%">'
        f'<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        f'<stop offset="1" stop-opacity=".1"/>'
        f"</linearGradient>"
        f'<clipPath id="r">'
        f'<rect width="{total_width}" height="{_HEIGHT}" rx="3" fill="#fff"/>'
        f"</clipPath>"
        f'<g clip-path="url(#r)">'
        f'<rect width="{_LABEL_WIDTH}" height="{_HEIGHT}" fill="#555"/>'
        f'<rect x="{_LABEL_WIDTH}" width="{val_width}" '
        f'height="{_HEIGHT}" fill="{color_safe}"/>'
        f'<rect width="{total_width}" height="{_HEIGHT}" fill="url(#s)"/>'
        f"</g>"
        f'<g fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        f'text-rendering="geometricPrecision" font-size="11">'
        f'<text x="{label_cx}" y="15" fill="#010101" '
        f'fill-opacity=".3">{label_safe}</text>'
        f'<text x="{label_cx}" y="14">{label_safe}</text>'
        f'<text x="{value_cx}" y="15" fill="#010101" '
        f'fill-opacity=".3">{value_safe}</text>'
        f'<text x="{value_cx}" y="14">{value_safe}</text>'
        f"</g>"
        f"</svg>"
    )


def _extract_robustness_score(run: dict) -> Optional[float]:
    """Extract robustness score (0.0 to 1.0) from a run dictionary."""
    metrics = run.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    for key in ("robustness_score", "auc_robustness", "robustness"):
        val = run.get(key)
        if val is None:
            val = metrics.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

    comparison = run.get("comparison") or metrics.get("comparison") or {}
    if isinstance(comparison, dict):
        rob_metric = comparison.get("metrics", {}).get("robustness", {})
        if isinstance(rob_metric, dict):
            auc = rob_metric.get("trained", {}).get("auc_robustness")
            if isinstance(auc, (int, float)) and not isinstance(auc, bool):
                return float(auc)
        for key in ("trained_auc", "auc_robustness", "robustness_score"):
            auc = comparison.get(key)
            if isinstance(auc, (int, float)) and not isinstance(auc, bool):
                return float(auc)

    trained_res = run.get("trained_results") or metrics.get("trained_results") or {}
    if isinstance(trained_res, dict):
        rob_res = trained_res.get("robustness", {})
        if isinstance(rob_res, dict):
            auc = rob_res.get("auc_robustness") or rob_res.get("score")
            if isinstance(auc, (int, float)) and not isinstance(auc, bool):
                return float(auc)
        elif isinstance(rob_res, (int, float)) and not isinstance(rob_res, bool):
            return float(rob_res)

    return None


def _get_latest_badge_color(score: float) -> str:
    """Return badge color based on score threshold: green (>80%), yellow (50-80%), red (<50%)."""
    if score > 0.80:
        return "#22c55e"  # green
    elif score >= 0.50:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


@router.get(
    "/latest.svg",
    summary="Render dynamic robustness badge from most recent run",
    responses={
        200: {"content": {"image/svg+xml": {}}},
    },
)
async def latest_badge_svg() -> Response:
    """Render dynamic SVG badge reflecting the robustness score of the most recent completed run."""
    try:
        from nightmarenet.pipeline_runner import list_all_runs

        all_runs = list_all_runs(include_historical=True)
    except Exception as e:
        logger.warning("Failed to retrieve runs for latest badge: %s", e)
        all_runs = []

    completed_runs = [r for r in all_runs if isinstance(r, dict) and r.get("status") == "complete"]
    completed_runs.sort(
        key=lambda r: float(r.get("last_heartbeat") or r.get("start_time") or 0.0),
        reverse=True,
    )

    score: Optional[float] = None
    for run in completed_runs:
        s = _extract_robustness_score(run)
        if s is not None:
            score = s
            break

    if score is None:
        svg = _render_svg("no data", "#9f9f9f")
    else:
        color = _get_latest_badge_color(score)
        svg = _render_svg(score, color)

    return Response(
        content=svg,
        media_type=_SVG_CONTENT_TYPE,
        headers={"Cache-Control": _CACHE_HEADER},
    )


@router.get(
    "/{score}.svg",
    summary="Render a robustness badge as SVG",
    responses={
        200: {"content": {"image/svg+xml": {}}},
        400: {"description": "Score must be a real number in [0, 1]"},
    },
)
async def badge_svg(score: float) -> Response:
    """Render a shields.io-style SVG badge for ``score`` ∈ ``[0, 1]``."""
    s = _validate_score(score)
    color, _label = _classify(s)
    svg = _render_svg(s, color)
    return Response(
        content=svg,
        media_type=_SVG_CONTENT_TYPE,
        headers={"Cache-Control": _CACHE_HEADER},
    )


@router.get(
    "/{score}.json",
    summary="Return badge metadata as JSON",
    responses={
        200: {"description": "Badge metadata"},
        400: {"description": "Score must be a real number in [0, 1]"},
    },
)
async def badge_json(score: float) -> Response:
    """Return JSON describing the badge so clients can render their own."""
    s = _validate_score(score)
    color, label = _classify(s)
    payload = {
        "score": round(s, 4),
        "color": color,
        "label": label,
        "message": f"{s:.2f}",
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        headers={"Cache-Control": _CACHE_HEADER},
    )
