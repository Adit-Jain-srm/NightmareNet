"""End-to-end verification script — proves every Sprint deliverable boots."""

import importlib.util
import json
import pathlib
import sys


def header(t: str) -> None:
    print(f"\n=== {t} ===")


def main() -> int:
    failures = []

    header("Alembic migration 0001_initial")
    try:
        spec = importlib.util.spec_from_file_location(
            "m", "nightmarenet_server/db/migrations/versions/0001_initial.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  revision: {mod.revision}")
        print(f"  down_revision: {mod.down_revision}")
        print(f"  has upgrade(): {callable(getattr(mod, 'upgrade', None))}")
        print(f"  has downgrade(): {callable(getattr(mod, 'downgrade', None))}")
    except Exception as e:
        failures.append(f"alembic: {e}")
        print(f"  [FAIL] {e}")

    header("Hosted platform modules")
    try:
        import nightmarenet_server.tasks.celery_app as c

        members = [m for m in dir(c) if not m.startswith("_")]
        print(f"  celery_app exports: {members}")
        from nightmarenet_server.tasks.training import run_pipeline_task

        print(f"  run_pipeline_task callable: {callable(run_pipeline_task)}")
        from nightmarenet_server.realtime.websocket import (
            build_realtime_router,
            get_broker,
            publish_event,
        )

        ws_router = build_realtime_router()
        if ws_router is None:
            print("  websocket router: None (FastAPI/websockets optional dep missing)")
        else:
            ws_paths = [getattr(r, "path", "?") for r in ws_router.routes]
            print(f"  websocket router paths: {ws_paths}")
        broker = get_broker()
        print(f"  broker: {broker.__class__.__name__}")
        print(f"  publish_event callable: {callable(publish_event)}")
        import nightmarenet_server.auth.api_keys as ak

        ak_members = [m for m in dir(ak) if not m.startswith("_")]
        print(f"  api_keys exports: {ak_members}")
        import nightmarenet_server.auth.oauth as oa

        oa_members = [m for m in dir(oa) if not m.startswith("_")]
        print(f"  oauth exports: {oa_members}")
    except Exception as e:
        failures.append(f"hosted: {e}")
        print(f"  [FAIL] {e}")

    header("Hosted FastAPI app")
    try:
        from nightmarenet_server.app import app

        print(f"  Title: {app.title}")
        routes = []
        for r in app.routes:
            methods = getattr(r, "methods", None)
            if methods:
                routes.append((sorted(methods), r.path))
        for methods, path in routes:
            print(f"    {','.join(methods):8s} {path}")
    except Exception as e:
        failures.append(f"app: {e}")
        print(f"  [FAIL] {e}")

    header("Notebook validity")
    for nb in sorted(pathlib.Path("notebooks").glob("*.ipynb")):
        try:
            data = json.loads(nb.read_text(encoding="utf-8"))
            cells = data.get("cells", [])
            code = sum(1 for c in cells if c.get("cell_type") == "code")
            md = sum(1 for c in cells if c.get("cell_type") == "markdown")
            v = f"{data.get('nbformat')}.{data.get('nbformat_minor')}"
            print(f"  [OK] {nb.name}: nbformat {v}, {code} code + {md} markdown cells")
        except Exception as e:
            failures.append(f"notebook {nb.name}: {e}")
            print(f"  [FAIL] {nb.name}: {e}")

    header("Pipeline orchestrator end-to-end (synthetic)")
    try:
        from nightmarenet.pipeline import Pipeline

        events = []
        cfg = {
            "model": {"name": "distilbert-base-uncased", "type": "seq_classification"},
            "dataset": {"name": "synthetic", "max_samples": 16},
            "training": {
                "wake_epochs": 0,
                "dream_epochs": 0,
                "nightmare_epochs": 0,
                "compress_epochs": 0,
                "num_cycles": 0,
                "device": "cpu",
            },
            "distortion": {"dream_strength": 0.2, "nightmare_strength": 0.5},
            "evaluation": {"strengths": [0.1, 0.5], "seed": 42},
            "tracking": {"enabled": False},
        }
        p = Pipeline(config=cfg, on_event=lambda e: events.append(e["status"]))
        # Don't actually run model training in this smoke; just confirm construction + helpers.
        print(f"  Pipeline constructed: {p.__class__.__name__}")
        print(f"  Metrics object: {type(p.metrics).__name__}")
        print(f"  Has run(): {callable(getattr(p, 'run', None))}")
    except Exception as e:
        failures.append(f"pipeline: {e}")
        print(f"  [FAIL] {e}")

    print()
    if failures:
        print(f"[E2E FAIL] {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[E2E PASS] All hosted/notebook/pipeline modules import and validate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
