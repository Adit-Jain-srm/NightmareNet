"""Unit tests for natural-language search query parsing."""

from datetime import datetime, timezone, tzinfo
from typing import Self

import pytest

from nightmarenet_server.search import query_parser
from nightmarenet_server.search.query_parser import ParsedQuery, parse_query


@pytest.mark.parametrize(
    ("query", "expected_status"),
    [
        ("completed runs", "completed"),
        ("complete runs", "completed"),
        ("FAILED runs", "failed"),
        ("running experiments", "running"),
        ("queued experiments", "queued"),
        ("pending jobs", "pending"),
    ],
)
def test_parse_query_extracts_and_normalizes_status(
    query: str, expected_status: str
) -> None:
    parsed = parse_query(query)

    assert parsed.filters["status"] == expected_status


@pytest.mark.parametrize(
    ("query", "expected_model"),
    [
        ("model DistilBERT", "distilbert"),
        ("using BERT-Large", "bert-large"),
        ("used org/model.v2", "org/model.v2"),
    ],
)
def test_parse_query_extracts_model_case_insensitively(
    query: str, expected_model: str
) -> None:
    parsed = parse_query(query)

    assert parsed.filters["model"] == expected_model


@pytest.mark.parametrize(
    ("query", "expected_metric"),
    [
        (
            "robustness > 0.7",
            {"field": "robustness", "op": ">", "value": 0.7},
        ),
        (
            "accuracy <= 0.95",
            {"field": "accuracy", "op": "<=", "value": 0.95},
        ),
        (
            "loss >= -1.5",
            {"field": "loss", "op": ">=", "value": -1.5},
        ),
        (
            "where nightmare strength = 0.8",
            {"field": "nightmare_strength", "op": "=", "value": 0.8},
        ),
    ],
)
def test_parse_query_extracts_metric_comparisons(
    query: str, expected_metric: dict[str, object]
) -> None:
    parsed = parse_query(query)

    assert parsed.filters["metrics"] == [expected_metric]


def test_parse_query_extracts_multiple_metric_filters_in_order() -> None:
    parsed = parse_query("accuracy >= 0.9 and loss < 0.2")

    assert parsed.filters["metrics"] == [
        {"field": "accuracy", "op": ">=", "value": 0.9},
        {"field": "loss", "op": "<", "value": 0.2},
    ]


@pytest.mark.parametrize(
    ("query", "expected_field"),
    [
        ("where robustness > 0.7", "robustness"),
        ("with nightmare strength > 0.7", "nightmare_strength"),
        ("having test loss < 1.0", "test_loss"),
        ("whose accuracy >= 0.8", "accuracy"),
    ],
)
def test_parse_query_strips_filter_prefixes_from_metric_names(
    query: str, expected_field: str
) -> None:
    parsed = parse_query(query)

    assert parsed.filters["metrics"][0]["field"] == expected_field


def test_parse_query_extracts_excluded_terms_case_insensitively() -> None:
    parsed = parse_query("not Char_Swap and NOT typo-noise")

    assert parsed.filters["exclude_terms"] == ["char_swap", "typo-noise"]


def test_parse_query_adds_created_after_for_last_week(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            assert tz == timezone.utc
            return cls(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(query_parser, "datetime", FixedDateTime)

    parsed = parse_query("completed runs from last week")

    assert parsed.filters["created_after"] == "2026-08-01T12:00:00+00:00"


def test_parse_query_last_week_matching_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(query_parser, "datetime", FixedDateTime)

    parsed = parse_query("LAST WEEK")

    assert parsed.filters["created_after"] == "2026-08-01T12:00:00+00:00"


def test_parse_query_preserves_trimmed_text() -> None:
    parsed = parse_query("   completed runs   ")

    assert parsed.text == "completed runs"


def test_parse_query_collects_lowercase_search_terms() -> None:
    parsed = parse_query("Robustness DistilBERT char_swap")

    assert parsed.terms == ["robustness", "distilbert", "char_swap"]


def test_parse_query_ignores_tokens_shorter_than_three_characters() -> None:
    parsed = parse_query("a to AI robust")

    assert parsed.terms == ["robust"]


def test_parse_query_empty_input_returns_empty_result() -> None:
    parsed = parse_query("   ")

    assert parsed == ParsedQuery(text="", filters={}, terms=[])


def test_parse_query_combines_supported_filters() -> None:
    parsed = parse_query(
        "completed runs using DistilBERT where robustness > 0.7 "
        "and not char_swap"
    )

    assert parsed.filters == {
        "status": "completed",
        "model": "distilbert",
        "metrics": [
            {"field": "robustness", "op": ">", "value": 0.7},
        ],
        "exclude_terms": ["char_swap"],
    }


def test_parse_query_returns_independent_filter_and_term_containers() -> None:
    first = parse_query("completed run")
    second = parse_query("failed run")

    first.filters["extra"] = True
    first.terms.append("extra")

    assert "extra" not in second.filters
    assert "extra" not in second.terms