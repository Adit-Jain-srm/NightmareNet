"""Tests for optional AI-generated failure-cluster summaries."""

from types import SimpleNamespace
from unittest.mock import Mock

from nightmarenet.evaluation.ai_summary import (
    append_ai_summary_to_report,
    build_failure_summary_prompt,
    generate_failure_cluster_summary,
)


def make_mock_client(summary: str = "Negation failures are dominant.") -> Mock:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=summary),
            )
        ]
    )

    client = Mock()
    client.chat.completions.create.return_value = response
    return client


def test_build_prompt_contains_cluster_data() -> None:
    clusters = {
        "negation_injection": {
            "failure_count": 6,
            "percentage": 60.0,
        },
        "synonym_substitution": {
            "failure_count": 4,
            "percentage": 40.0,
        },
    }

    prompt = build_failure_summary_prompt(clusters)

    assert "negation_injection" in prompt
    assert "synonym_substitution" in prompt
    assert "60.0" in prompt


def test_generate_summary_with_mock_client() -> None:
    client = make_mock_client()

    summary = generate_failure_cluster_summary(
        {"negation_injection": {"failure_count": 6}},
        client=client,
    )

    assert summary == "Negation failures are dominant."
    client.chat.completions.create.assert_called_once()


def test_empty_clusters_return_none() -> None:
    client = make_mock_client()

    summary = generate_failure_cluster_summary({}, client=client)

    assert summary is None
    client.chat.completions.create.assert_not_called()


def test_api_failure_returns_none() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")

    summary = generate_failure_cluster_summary(
        {"character_noise": {"failure_count": 3}},
        client=client,
    )

    assert summary is None


def test_empty_response_returns_none() -> None:
    client = make_mock_client("   ")

    summary = generate_failure_cluster_summary(
        {"synonym_substitution": {"failure_count": 2}},
        client=client,
    )

    assert summary is None


def test_append_summary_to_report() -> None:
    report = "# Robustness Evaluation\n\nStandard metrics."
    summary = "Negation injection caused most failures."

    result = append_ai_summary_to_report(report, summary)

    assert "## AI-Generated Failure Cluster Summary" in result
    assert summary in result
    assert "Standard metrics." in result


def test_report_unchanged_without_summary() -> None:
    report = "# Robustness Evaluation\n"

    result = append_ai_summary_to_report(report, None)

    assert result == report