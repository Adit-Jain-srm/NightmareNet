"""Backfill the experiment search index from the hosted database."""

import argparse
import logging
import os
from typing import Any, List

from nightmarenet_server.search.embedder import ExperimentEmbedder, document_from_orm
from nightmarenet_server.search.index import SearchIndex

logger = logging.getLogger(__name__)


def reindex(database_url: str, index_path: str = "", backend: str = "faiss") -> int:
    from nightmarenet_server.models import AuditLog, Run, get_session_factory

    session_factory = get_session_factory(database_url)
    session = session_factory()
    embedder = ExperimentEmbedder()
    index = SearchIndex(backend=backend, path=index_path or None)
    count = 0
    try:
        runs: List[Any] = session.query(Run).all()
        for run in runs:
            try:
                experiment = getattr(run, "experiment", None)
                audit_logs = []
                if experiment is not None:
                    audit_logs = (
                        session.query(AuditLog)
                        .filter(AuditLog.resource_id.in_([run.id, experiment.id]))
                        .all()
                    )
                doc = document_from_orm(run, experiment=experiment, audit_logs=audit_logs)
                index.add(doc.run_id, embedder.embed_run(doc), doc.metadata())
                count += 1
            except Exception:
                logger.exception("Failed to index run %s; continuing", getattr(run, "id", ""))
    finally:
        session.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the NightmareNet experiment search index."
    )
    parser.add_argument("--database-url", default=os.environ.get("NIGHTMARENET_DATABASE_URL"))
    parser.add_argument("--index-path", default=os.environ.get("NIGHTMARENET_SEARCH_INDEX", ""))
    parser.add_argument("--backend", default=os.environ.get("NIGHTMARENET_SEARCH_BACKEND", "faiss"))
    args = parser.parse_args()
    if not args.database_url:
        from nightmarenet_server.models.base import DEFAULT_DATABASE_URL

        args.database_url = DEFAULT_DATABASE_URL
    count = reindex(args.database_url, index_path=args.index_path, backend=args.backend)
    print(f"Indexed {count} runs")


if __name__ == "__main__":
    main()
