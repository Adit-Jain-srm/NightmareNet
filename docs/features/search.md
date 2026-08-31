# Semantic Experiment Search Guide

The hosted NightmareNet server can index every experiment run and search it with natural language. Queries are embedded into a vector space, matched against a FAISS index, and combined with lightweight structured filters parsed straight from the query text (status, model, metric thresholds, and more).

## Overview

Search lives in `nightmarenet_server/search/`:

| Component | Source | Responsibility |
|---|---|---|
| `parse_query` | `query_parser.py` | Extracts structured filters from natural-language text. |
| `ExperimentEmbedder` | `embedder.py` | Embeds runs and queries into 384-dim vectors. |
| `SearchIndex` | `index.py` | FAISS-backed vector index with a NumPy fallback and disk persistence. |
| `build_search_router` | `endpoints.py` | FastAPI router exposing `POST /api/v1/search`. |
| `reindex` | `reindex.py` | Backfills the index from the hosted database. |

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) when `sentence-transformers` is installed. When it is not, a deterministic hashing embedder keeps search usable without downloading model weights.

---

## Quick Start

### Query via the API

The router is mounted at `POST /api/v1/search`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "completed distilbert runs with robustness > 0.6 last week", "top_k": 10}'
```

The response contains ranked results plus the filters that were parsed from the query:

```json
{
  "results": [
    {"run_id": "...", "relevance_score": 0.87, "summary": "...", "metadata": {}}
  ],
  "filters": {"status": "completed", "model": "distilbert", "metrics": [{"field": "robustness", "op": ">", "value": 0.6}]},
  "backend": "faiss"
}
```

### Build the index

Backfill the index from the hosted database:

```bash
python -m nightmarenet_server.search.reindex \
    --database-url "$NIGHTMARENET_DATABASE_URL" \
    --index-path .nightmarenet_search \
    --backend faiss
```

---

## Query Syntax

`parse_query` recognizes the following patterns inside otherwise free-form text:

| Filter | Trigger | Example phrase |
|---|---|---|
| `status` | `completed`, `complete`, `failed`, `running`, `queued`, `pending` | "failed runs" |
| `model` | `model / using / used <name>` | "using distilbert-base-uncased" |
| `metrics` | `<field> <op> <value>` where op is `>`, `>=`, `<`, `<=`, `=` | "robustness > 0.6" |
| `created_after` | `last week` | "trained last week" |
| `exclude_terms` | `not <term>` | "not cifar10" |

Everything else becomes free-text `terms` that feed the semantic embedding. Structured filters are applied by `SearchIndex.hybrid_search`, which only ranks candidates whose metadata matches the parsed filters.

---

## Configuration

Search behaviour is controlled through environment variables:

| Variable | Default | Description |
|---|---|---|
| `NIGHTMARENET_SEARCH_BACKEND` | `faiss` | Index backend. Any value other than `faiss` uses the NumPy fallback. |
| `NIGHTMARENET_SEARCH_INDEX` | `.nightmarenet_search` | Directory where the persisted index (`index.json`) is written. |
| `NIGHTMARENET_DATABASE_URL` | — | Source database for `reindex` (also accepted via `--database-url`). |

> [!NOTE]
> FAISS and `sentence-transformers` are optional. Without FAISS, `SearchIndex` falls back to NumPy cosine ranking; without `sentence-transformers`, `ExperimentEmbedder` falls back to a deterministic hashing embedding. Both fallbacks keep the API functional.

---

## API Reference

### `parse_query`

```python
from nightmarenet_server.search.query_parser import parse_query

parsed = parse_query("completed distilbert runs with robustness > 0.6 last week")
# ParsedQuery(text=..., filters={...}, terms=[...])
```

### `ExperimentEmbedder`

```python
ExperimentEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384, model=None)
```

| Method | Returns | Description |
|---|---|---|
| `embed_query(query)` | `np.ndarray` | Embeds a query string. |
| `embed_run(run)` | `np.ndarray` | Embeds an `ExperimentDocument`. |
| `serialize_run(run)` | `str` | Flattens a run's config, metrics, events, and audit logs into text. |

The module also exposes `EMBEDDING_DIM = 384` and the `ExperimentDocument` dataclass used to describe a run.

### `SearchIndex`

```python
SearchIndex(backend="faiss", path=None, dimension=EMBEDDING_DIM)
```

| Method | Description |
|---|---|
| `add(run_id, embedding, metadata)` | Adds/updates a vector and persists the index. |
| `delete(run_id)` | Removes a run and persists. |
| `search(query_embedding, top_k=10)` | Pure vector search. Returns `list[SearchHit]`. |
| `hybrid_search(query_embedding, filters=None, top_k=10)` | Vector search filtered by parsed metadata. |
| `persist()` | Atomically writes `index.json` to `path`. |

`SearchHit` is a dataclass with `run_id`, `score`, and `metadata`.

### `build_search_router`

```python
from nightmarenet_server.search.endpoints import build_search_router

router = build_search_router()   # returns a FastAPI APIRouter, or None if FastAPI is unavailable
```

The request body accepts `query` (1–1000 chars), `top_k` (1–50, default 10), and optional `filters`. Body filters are merged over the filters parsed from the query.

### `reindex`

```python
reindex(database_url: str, index_path: str = "", backend: str = "faiss") -> int
```

Iterates over every `Run` in the database, builds an `ExperimentDocument`, embeds it, and adds it to the index. Returns the number of runs indexed.

---

## Examples

### Programmatic search

```python
from nightmarenet_server.search.embedder import ExperimentEmbedder
from nightmarenet_server.search.index import SearchIndex
from nightmarenet_server.search.query_parser import parse_query

embedder = ExperimentEmbedder()
index = SearchIndex(backend="faiss")

parsed = parse_query("failed runs not cifar10")
embedding = embedder.embed_query(parsed.text)
hits = index.hybrid_search(embedding, filters=parsed.filters, top_k=5)
for hit in hits:
    print(hit.run_id, hit.score)
```

### Index a single run

```python
from nightmarenet_server.search.embedder import ExperimentDocument, ExperimentEmbedder
from nightmarenet_server.search.index import SearchIndex

doc = ExperimentDocument(run_id="run-123", name="sst2-v1", model="distilbert-base-uncased",
                         status="completed", metrics={"robustness": 0.68})
embedder = ExperimentEmbedder()
index = SearchIndex()
index.add(doc.run_id, embedder.embed_run(doc), doc.metadata())
```

---

## Related Documentation

- [Webhook Notifications](webhooks.md) — react to run completion and regression events.
- [Getting Started](../tutorials/getting-started.md) — install and run your first cycle.
