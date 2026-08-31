"""Unit tests for nightmarenet_server.search (hosted semantic search)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nightmarenet_server.search import ExperimentDocument, ExperimentEmbedder, SearchIndex
from nightmarenet_server.search.embedder import document_from_orm
from nightmarenet_server.search.query_parser import parse_query


def test_document_from_orm_maps_run_fields() -> None:
    run = SimpleNamespace(
        id="run-42",
        experiment_id="exp-9",
        status="completed",
        phase="nightmare",
        metrics_json='{"robustness": 0.81}',
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T01:00:00Z",
        events=[],
    )
    experiment = SimpleNamespace(
        name="sst2 sweep",
        config_json='{"model_name": "distilbert-base-uncased"}',
        created_at="2025-12-31T00:00:00Z",
    )

    doc = document_from_orm(run, experiment=experiment)

    assert doc.run_id == "run-42"
    assert doc.experiment_id == "exp-9"
    assert doc.name == "sst2 sweep"
    assert doc.model == "distilbert-base-uncased"
    assert doc.metrics["robustness"] == 0.81


def test_document_from_orm_includes_audit_logs() -> None:
    run = SimpleNamespace(
        id="run-1",
        experiment_id="exp-1",
        status="running",
        phase="dream",
        metrics_json="{}",
        started_at="",
        completed_at="",
        events=[],
    )
    log = SimpleNamespace(
        action="update",
        resource_type="run",
        resource_id="run-1",
        metadata_json='{"note": "phase change"}',
        timestamp="2026-01-02T12:00:00Z",
    )

    doc = document_from_orm(run, audit_logs=[log])

    assert doc.audit_logs[0]["action"] == "update"
    assert doc.audit_logs[0]["metadata"]["note"] == "phase change"


def test_embedder_batch_runs_share_dimension() -> None:
    embedder = ExperimentEmbedder(model=False)
    docs = [
        ExperimentDocument(run_id="a", name="alpha", metrics={"robustness": 0.5}),
        ExperimentDocument(run_id="b", name="beta", metrics={"robustness": 0.9}),
    ]

    vectors = [embedder.embed_run(doc) for doc in docs]

    assert all(vec.shape == (384,) for vec in vectors)
    assert not np.allclose(vectors[0], vectors[1])


def test_embedder_uses_injected_model() -> None:
    calls = []

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            calls.append(list(texts))
            return [np.full(384, 0.5, dtype=np.float32)]

    embedder = ExperimentEmbedder(model=FakeModel())
    out = embedder.embed_query("robustness drop")

    assert out.shape == (384,)
    assert calls == [["robustness drop"]]


def test_empty_index_returns_no_hits(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path / "empty"), backend="numpy")
    query = np.zeros(384, dtype=np.float32)

    assert index.search(query) == []


def test_search_ranks_by_similarity(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path / "rank"), backend="numpy")
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1.0

    close = np.zeros(384, dtype=np.float32)
    close[0] = 0.9
    close[1] = 0.1
    close /= np.linalg.norm(close)

    far = np.zeros(384, dtype=np.float32)
    far[2] = 1.0

    index.add("near", close, {"status": "completed"})
    index.add("far", far, {"status": "completed"})

    hits = index.search(query, top_k=2)

    assert [hit.run_id for hit in hits] == ["near", "far"]
    assert hits[0].score > hits[1].score


def test_parse_query_empty_string() -> None:
    parsed = parse_query("   ")

    assert parsed.text == ""
    assert parsed.filters == {}
    assert parsed.terms == []


def test_parse_query_extracts_model_filter() -> None:
    parsed = parse_query("runs using DistilBERT with high robustness")

    assert parsed.filters["model"] == "distilbert"


def test_parse_query_last_week_date_filter() -> None:
    parsed = parse_query("completed experiments from last week")

    assert parsed.filters["status"] == "completed"
    assert "created_after" in parsed.filters


def test_parse_query_handles_quotes_and_brackets() -> None:
    parsed = parse_query('status:running model "bert-base" (nightmare)')

    assert parsed.text.startswith("status:running")
    assert "status" in parsed.terms or "running" in parsed.terms


def test_reindex_indexes_all_runs(monkeypatch, tmp_path) -> None:
    pytest.importorskip("sqlalchemy")
    from nightmarenet_server.models import (
        Experiment,
        Org,
        Project,
        Run,
        get_session_factory,
        init_db,
    )
    from nightmarenet_server.search import reindex as reindex_module

    db_url = f"sqlite:///{tmp_path / 'all-runs.db'}"
    init_db(db_url)
    session = get_session_factory(db_url)()
    try:
        session.add_all(
            [
                Org(id="org-1", name="Org"),
                Project(id="proj-1", org_id="org-1", name="Proj"),
                Experiment(id="exp-1", project_id="proj-1", name="Exp", config_json="{}"),
                Run(id="run-a", experiment_id="exp-1", status="completed"),
                Run(id="run-b", experiment_id="exp-1", status="failed"),
            ]
        )
        session.commit()
    finally:
        session.close()

    class DummyEmbedder:
        def embed_run(self, doc: object) -> np.ndarray:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1.0
            return vector

    monkeypatch.setattr(reindex_module, "ExperimentEmbedder", lambda: DummyEmbedder())

    count = reindex_module.reindex(
        db_url,
        index_path=str(tmp_path / "idx"),
        backend="numpy",
    )

    assert count == 2
    loaded = SearchIndex(path=str(tmp_path / "idx"), backend="numpy")
    assert len(loaded.search(np.zeros(384, dtype=np.float32), top_k=10)) == 2


def test_reindex_continues_when_one_run_fails(monkeypatch, tmp_path) -> None:
    pytest.importorskip("sqlalchemy")
    from nightmarenet_server.models import (
        Experiment,
        Org,
        Project,
        Run,
        get_session_factory,
        init_db,
    )
    from nightmarenet_server.search import reindex as reindex_module

    db_url = f"sqlite:///{tmp_path / 'partial.db'}"
    init_db(db_url)
    session = get_session_factory(db_url)()
    try:
        session.add_all(
            [
                Org(id="org-1", name="Org"),
                Project(id="proj-1", org_id="org-1", name="Proj"),
                Experiment(id="exp-1", project_id="proj-1", name="Exp", config_json="{}"),
                Run(id="bad", experiment_id="exp-1", status="completed"),
                Run(id="good", experiment_id="exp-1", status="completed"),
            ]
        )
        session.commit()
    finally:
        session.close()

    original = reindex_module.document_from_orm

    def flaky_document(run: object, **kwargs: object) -> object:
        if getattr(run, "id", "") == "bad":
            raise ValueError("bad row")
        return original(run, **kwargs)

    class DummyEmbedder:
        def embed_run(self, doc: object) -> np.ndarray:
            vector = np.zeros(384, dtype=np.float32)
            vector[1] = 1.0
            return vector

    monkeypatch.setattr(reindex_module, "ExperimentEmbedder", lambda: DummyEmbedder())
    monkeypatch.setattr(reindex_module, "document_from_orm", flaky_document)

    count = reindex_module.reindex(db_url, index_path=str(tmp_path / "idx2"), backend="numpy")

    assert count == 1
