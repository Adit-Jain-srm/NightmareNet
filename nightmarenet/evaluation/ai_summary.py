"""Optional AI-generated summaries for robustness failure clusters."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an AI robustness evaluation assistant. "
    "Summarize only the provided failure-cluster statistics. "
    "Do not invent causes, values, examples, or recommendations."
)


def _normalize_clusters(
    clusters: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert supported cluster data into a serializable list."""
    if isinstance(clusters, Mapping):
        normalized: list[dict[str, Any]] = []

        for cluster_name, cluster_data in clusters.items():
            if isinstance(cluster_data, Mapping):
                normalized.append(
                    {
                        "cluster": str(cluster_name),
                        **dict(cluster_data),
                    }
                )
            else:
                normalized.append(
                    {
                        "cluster": str(cluster_name),
                        "value": cluster_data,
                    }
                )

        return normalized

    return [dict(cluster) for cluster in clusters]


def build_failure_summary_prompt(
    clusters: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> str:
    """Create a grounded prompt from failure-cluster statistics."""
    normalized = _normalize_clusters(clusters)

    return (
        "Generate a concise summary of these robustness failure clusters.\n\n"
        "Requirements:\n"
        "- Identify the dominant failure patterns.\n"
        "- Explain observed weaknesses only from the supplied data.\n"
        "- Mention confidence or perplexity changes only when present.\n"
        "- Provide no more than three practical recommendations.\n"
        "- Do not invent missing information.\n\n"
        "Failure cluster data:\n"
        f"{json.dumps(normalized, indent=2, default=str)}"
    )


def generate_failure_cluster_summary(
    clusters: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_AI_MODEL,
    client: Any | None = None,
) -> str | None:
    """Generate an optional AI summary.

    Returns None when dependency, configuration, data, or API access
    is unavailable, allowing normal evaluation to continue.
    """
    normalized = _normalize_clusters(clusters)

    if not normalized:
        logger.info(
            "Skipping AI summary because no failure clusters were provided."
        )
        return None

    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY is not configured. "
                "Continuing without an AI-generated summary."
            )
            return None

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning(
                "The optional OpenAI dependency is unavailable. "
                "Install NightmareNet with the [ai] extra."
            )
            return None

        client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_failure_summary_prompt(normalized),
                },
            ],
        )

        summary = response.choices[0].message.content

        if not summary or not summary.strip():
            logger.warning(
                "AI summary generation returned an empty response."
            )
            return None

        return summary.strip()

    except Exception as exc:
        logger.warning(
            "AI summary generation failed: %s. "
            "Continuing with the standard report.",
            exc,
        )
        return None


def append_ai_summary_to_report(
    report: str,
    summary: str | None,
) -> str:
    """Append an optional AI summary section to a Markdown report."""
    if not summary:
        return report

    separator = "" if report.endswith("\n") else "\n"

    return (
        f"{report}{separator}\n"
        "## AI-Generated Failure Cluster Summary\n\n"
        f"{summary}\n"
    )