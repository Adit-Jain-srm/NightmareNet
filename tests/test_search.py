from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nightmarenet_server.search.embedder import ExperimentDocument, ExperimentEmbedder
from nightmarenet_server.search.index import SearchIndex
from nightmarenet_server.search.query_parser import parse_query


def test_embedding_determinism_without_model() -> None:
    embedder = ExperimentEmbedder(model=False)
    doc = ExperimentDocument(
        run_id="exp-1",
        name="synonym stress",
        config={"distortions": ["synonym"], "nightmare_strength": 0.8},
        metrics={"robustness": 0.72},
    )

    first = embedder.embed_run(doc)
    second = embedder.embed_run(doc)

    assert first.shape == (384,)
    assert np.allclose(first, second)


def test_index_crud_and_persistence(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path), backend="numpy")
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1

    index.add("run-1", vector, {"status": "completed"})
    assert index.search(vector, top_k=1)[0].run_id == "run-1"

    loaded = SearchIndex(path=str(tmp_path), backend="numpy")
    assert loaded.search(vector, top_k=1)[0].metadata["status"] == "completed"

    loaded.delete("run-1")
    assert loaded.search(vector) == []


def test_index_starts_empty_when_persisted_file_is_corrupt(tmp_path) -> None:
    (tmp_path / "index.json").write_text("{not-json", encoding="utf-8")

    index = SearchIndex(path=str(tmp_path), backend="numpy")

    assert index.search(np.zeros(384, dtype=np.float32)) == []


def test_hybrid_filtering(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path), backend="numpy")
    query = np.zeros(384, dtype=np.float32)
    query[2] = 1
    index.add(
        "good",
        query,
        {
            "status": "completed",
            "model": "DistilBERT",
            "metrics": {"robustness": 0.82},
            "config": {"nightmare_strength": 0.8},
        },
    )
    other = np.zeros(384, dtype=np.float32)
    other[3] = 1
    index.add(
        "bad",
        other,
        {"status": "failed", "model": "GPT-2", "metrics": {"robustness": 0.2}},
    )

    hits = index.hybrid_search(
        query,
        filters={
            "status": "completed",
            "metrics": [{"field": "robustness", "op": ">", "value": 0.7}],
        },
    )

    assert [hit.run_id for hit in hits] == ["good"]

    config_hits = index.hybrid_search(
        query,
        filters={"metrics": [{"field": "nightmare_strength", "op": ">", "value": 0.7}]},
    )
    assert [hit.run_id for hit in config_hits] == ["good"]


def test_hybrid_filtering_filters_before_ranking(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path), backend="numpy")
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1

    for i in range(8):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0 - (i * 0.01)
        vector[1] = i * 0.01
        vector = vector / np.linalg.norm(vector)
        index.add(f"completed-{i}", vector, {"status": "completed"})

    failed = np.zeros(384, dtype=np.float32)
    failed[0] = 0.2
    failed[2] = 0.8
    failed = failed / np.linalg.norm(failed)
    index.add("failed-match", failed, {"status": "failed"})

    hits = index.hybrid_search(query, filters={"status": "failed"}, top_k=1)

    assert [hit.run_id for hit in hits] == ["failed-match"]


def test_index_supports_concurrent_access(tmp_path) -> None:
    index = SearchIndex(path=str(tmp_path), backend="numpy")
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1

    def add_and_search(i: int) -> None:
        vector = np.zeros(384, dtype=np.float32)
        vector[i % 384] = 1
        index.add(f"run-{i}", vector, {"status": "completed"})
        index.search(query, top_k=5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add_and_search, range(24)))

    assert len(index.search(query, top_k=50)) == 24


def test_query_parser_extracts_filters() -> None:
    parsed = parse_query("completed runs where nightmare strength > 0.7 and not char_swap")

    assert parsed.filters["status"] == "completed"
    assert parsed.filters["exclude_terms"] == ["char_swap"]
    assert parsed.filters["metrics"][0] == {
        "field": "nightmare_strength",
        "op": ">",
        "value": 0.7,
    }


def test_search_endpoint_returns_ranked_results(monkeypatch, tmp_path) -> None:
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    from nightmarenet_server.search import endpoints as search_endpoints

    index = SearchIndex(path=str(tmp_path), backend="numpy")
    vector = np.zeros(384, dtype=np.float32)
    vector[5] = 1
    index.add(
        "exp-47",
        vector,
        {
            "run_id": "exp-47",
            "name": "char-swap robustness",
            "model": "DistilBERT",
            "status": "completed",
            "metrics": {"robustness": 0.91},
        },
    )

    class DummyEmbedder:
        def embed_query(self, query: str) -> np.ndarray:
            return vector

    monkeypatch.setattr(search_endpoints, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(search_endpoints, "get_index", lambda: index)

    app = fastapi.FastAPI()
    app.include_router(search_endpoints.build_search_router())
    client = testclient.TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={"query": "completed runs with robustness > 0.8", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["run_id"] == "exp-47"
    assert body["results"][0]["relevance_score"] > 0
    assert body["filters"]["status"] == "completed"


def test_search_endpoint_hides_internal_errors(monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    from nightmarenet_server.search import endpoints as search_endpoints

    class BrokenEmbedder:
        def embed_query(self, query: str) -> np.ndarray:
            raise RuntimeError("secret internal path /tmp/index.faiss")

    monkeypatch.setattr(search_endpoints, "get_embedder", lambda: BrokenEmbedder())

    app = fastapi.FastAPI()
    app.include_router(search_endpoints.build_search_router())
    client = testclient.TestClient(app)

    response = client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."


def test_reindex_continues_after_bad_run(monkeypatch, tmp_path) -> None:
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

    db_url = f"sqlite:///{tmp_path / 'runs.db'}"
    init_db(db_url)
    session_factory = get_session_factory(db_url)
    session = session_factory()
    try:
        org = Org(id="org-1", name="Org")
        project = Project(id="project-1", org_id="org-1", name="Project")
        experiment = Experiment(
            id="experiment-1",
            project_id="project-1",
            name="Experiment",
            config_json="{}",
        )
        session.add_all(
            [
                org,
                project,
                experiment,
                Run(id="bad-run", experiment_id="experiment-1", status="completed"),
                Run(id="good-run", experiment_id="experiment-1", status="completed"),
            ]
        )
        session.commit()
    finally:
        session.close()

    class DummyEmbedder:
        def embed_run(self, doc: object) -> np.ndarray:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1
            return vector

    original_document_from_orm = reindex_module.document_from_orm

    def document_from_orm(run: object, **kwargs: object) -> object:
        if getattr(run, "id", "") == "bad-run":
            raise ValueError("bad row")
        return original_document_from_orm(run, **kwargs)

    monkeypatch.setattr(reindex_module, "ExperimentEmbedder", lambda: DummyEmbedder())
    monkeypatch.setattr(reindex_module, "document_from_orm", document_from_orm)

    count = reindex_module.reindex(
        db_url,
        index_path=str(tmp_path / "search-index"),
        backend="numpy",
    )

    assert count == 1
