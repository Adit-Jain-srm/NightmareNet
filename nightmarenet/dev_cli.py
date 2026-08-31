"""Unified developer task runner: ``nightmarenet dev <command>``.

Mirrors CI / Makefile targets using cross-platform ``subprocess`` calls
(Windows PowerShell, macOS, and Linux). Prefer this over remembering
individual tool invocations; Makefile targets remain supported.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _die_missing(tool: str, hint: str) -> int:
    print(f"error: {tool} not found. {hint}", file=sys.stderr)
    return 127


def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> int:
    """Run a command; return exit code. Never shells out through bash."""
    print(f"+ {' '.join(cmd)}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Ensure repo root is on PYTHONPATH like Makefile / CI
    py_path = merged.get("PYTHONPATH", "")
    root = str(REPO_ROOT)
    if root not in py_path.split(os.pathsep):
        merged["PYTHONPATH"] = root + (os.pathsep + py_path if py_path else "")
    try:
        completed = subprocess.run(  # noqa: S603 — intentional controlled argv
            list(cmd),
            cwd=str(cwd or REPO_ROOT),
            env=merged,
            check=False,
        )
    except FileNotFoundError:
        print(f"error: failed to execute {cmd[0]!r}", file=sys.stderr)
        return 127
    if check and completed.returncode != 0:
        return completed.returncode
    return completed.returncode


def cmd_lint(args: argparse.Namespace) -> int:
    """Run ruff + mypy (+ frontend ESLint unless --python-only)."""
    if not _which("ruff"):
        return _die_missing("ruff", "Install with: pip install -e '.[dev]'")
    if not _which("mypy"):
        return _die_missing("mypy", "Install with: pip install -e '.[dev]'")

    code = _run(["ruff", "check", "nightmarenet/", "scripts/", "tests/"])
    if code != 0:
        return code

    code = _run(
        [
            "mypy",
            "nightmarenet/",
            "--ignore-missing-imports",
            "--disable-error-code",
            "import-untyped",
            "--disable-error-code",
            "operator",
            "--python-version",
            "3.12",
        ]
    )
    if code != 0:
        return code

    if getattr(args, "python_only", False):
        return 0

    if not _which("npm"):
        return _die_missing("npm", "Node.js not found, install from https://nodejs.org/")
    return _run(["npm", "run", "lint"], cwd=REPO_ROOT / "frontend")


def cmd_test(args: argparse.Namespace) -> int:
    """Run pytest (CI markers) and optionally frontend vitest."""
    if getattr(args, "frontend", False):
        if not _which("npm"):
            return _die_missing("npm", "Node.js not found, install from https://nodejs.org/")
        return _run(["npm", "run", "test"], cwd=REPO_ROOT / "frontend")

    if not _which("pytest"):
        return _die_missing("pytest", "Install with: pip install -e '.[dev]'")

    cmd = [
        "pytest",
        "-m",
        "not slow",
        "--cov=nightmarenet",
        "--cov-report=xml",
        "--cov-report=term-missing",
    ]
    if getattr(args, "marker", None):
        cmd[2] = args.marker
    if getattr(args, "pytest_args", None):
        cmd.extend(args.pytest_args)
    return _run(cmd)


def cmd_format(args: argparse.Namespace) -> int:
    """Run ruff format (+ prettier when available)."""
    if not _which("ruff"):
        return _die_missing("ruff", "Install with: pip install -e '.[dev]'")
    code = _run(["ruff", "format", "."])
    if code != 0:
        return code

    if getattr(args, "python_only", False):
        return 0

    npm = _which("npm")
    if npm is None:
        print("note: npm not found; skipped prettier (Python formatting done).")
        return 0
    # Prefer project script if present; otherwise npx prettier when configured
    frontend = REPO_ROOT / "frontend"
    prettier_name = "prettier.cmd" if os.name == "nt" else "prettier"
    prettier = frontend / "node_modules" / ".bin" / prettier_name
    if prettier.exists():
        return _run([str(prettier), "--write", "."], cwd=frontend)
    print("note: prettier not installed in frontend/; skipped.")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Run Alembic migrations to head."""
    if not _which("alembic"):
        return _die_missing(
            "alembic",
            "Install hosted extras: pip install -e '.[hosted]' (or pip install alembic)",
        )
    revision = getattr(args, "revision", None) or "head"
    return _run(["alembic", "upgrade", revision])


def cmd_serve(args: argparse.Namespace) -> int:
    """Start API and frontend dev servers (cross-platform)."""
    if not _which("uvicorn"):
        return _die_missing("uvicorn", "Install with: pip install -e '.[api]'")
    if not getattr(args, "api_only", False) and not _which("npm"):
        return _die_missing("npm", "Node.js not found, install from https://nodejs.org/")

    procs: List[subprocess.Popen] = []
    env = os.environ.copy()
    root = str(REPO_ROOT)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "nightmarenet.api.app:app",
        "--reload",
        "--host",
        getattr(args, "host", "127.0.0.1"),
        "--port",
        str(getattr(args, "port", 8000)),
    ]
    print(f"+ {' '.join(api_cmd)}")
    procs.append(subprocess.Popen(api_cmd, cwd=str(REPO_ROOT), env=env))  # noqa: S603

    if not getattr(args, "api_only", False):
        fe_cmd = ["npm", "run", "dev"]
        print(f"+ (frontend) {' '.join(fe_cmd)}")
        procs.append(subprocess.Popen(fe_cmd, cwd=str(REPO_ROOT / "frontend"), env=env))  # noqa: S603

    def _shutdown(*_args: object) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            for proc in procs:
                code = proc.poll()
                if code is not None:
                    _shutdown()
                    return code if code != 0 else 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown()
        return 130


def cmd_docker(args: argparse.Namespace) -> int:
    """Bring up docker compose and verify API health."""
    docker = _which("docker")
    if docker is None:
        return _die_missing("docker", "Install Docker Desktop / Engine from https://docker.com/")

    compose = ["docker", "compose"]
    # Prefer `docker compose` (v2); fall back message if plugin missing handled by docker itself
    up_cmd = compose + ["up", "-d"]
    if getattr(args, "profile", None):
        up_cmd = compose + ["--profile", args.profile, "up", "-d"]
    if getattr(args, "build", False):
        up_cmd.append("--build")
    code = _run(up_cmd)
    if code != 0:
        return code

    if getattr(args, "no_health", False):
        return 0

    health_url = getattr(args, "health_url", "http://127.0.0.1:8000/api/v1/health")
    print(f"Waiting for health: {health_url}")
    deadline = time.time() + getattr(args, "timeout", 90)
    last_err = ""
    while time.time() < deadline:
        try:
            import urllib.request

            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    print("health check passed")
                    return 0
                last_err = f"status {resp.status}"
        except Exception as exc:  # noqa: BLE001 — surface connection errors
            last_err = str(exc)
        time.sleep(2)
    print(f"error: health check failed ({last_err})", file=sys.stderr)
    return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the standard local performance / robustness benchmark script."""
    script = REPO_ROOT / "scripts" / "run_benchmark.py"
    if not script.exists():
        print(f"error: benchmark script missing: {script}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script)]
    if getattr(args, "benchmark_args", None):
        cmd.extend(args.benchmark_args)
    return _run(cmd)


def cmd_check(args: argparse.Namespace) -> int:
    """Lint + typecheck + test (mirrors ``make check`` / CI core)."""
    lint_ns = argparse.Namespace(python_only=True)
    code = cmd_lint(lint_ns)
    if code != 0:
        return code
    test_ns = argparse.Namespace(frontend=False, marker=None, pytest_args=None)
    return cmd_test(test_ns)


def register_dev_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach the ``dev`` command group to the main nightmarenet parser."""
    dev = subparsers.add_parser(
        "dev",
        help="Unified developer tasks (lint, test, format, migrate, serve, …)",
        description=(
            "Cross-platform developer CLI. Prefer `nightmarenet dev <cmd>` over "
            "remembering individual tools. Makefile targets remain available."
        ),
    )
    dev_sub = dev.add_subparsers(dest="dev_command", required=True)

    p_lint = dev_sub.add_parser(
        "lint",
        help="Run ruff + mypy (+ frontend ESLint)",
    )
    p_lint.add_argument(
        "--python-only",
        action="store_true",
        help="Skip frontend ESLint",
    )
    p_lint.set_defaults(dev_handler=cmd_lint)

    p_test = dev_sub.add_parser("test", help="Run pytest with CI markers")
    p_test.add_argument(
        "--frontend",
        action="store_true",
        help="Run frontend vitest instead of pytest",
    )
    p_test.add_argument(
        "-m",
        "--marker",
        dest="marker",
        default=None,
        help='Pytest -m expression (default: "not slow")',
    )
    p_test.add_argument(
        "pytest_args",
        nargs="*",
        help="Extra args forwarded to pytest",
    )
    p_test.set_defaults(dev_handler=cmd_test)

    p_fmt = dev_sub.add_parser("format", help="Run ruff format (+ prettier if present)")
    p_fmt.add_argument("--python-only", action="store_true", help="Skip prettier")
    p_fmt.set_defaults(dev_handler=cmd_format)

    p_mig = dev_sub.add_parser("migrate", help="Run alembic upgrade")
    p_mig.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="Alembic revision (default: head)",
    )
    p_mig.set_defaults(dev_handler=cmd_migrate)

    p_serve = dev_sub.add_parser("serve", help="Start API + frontend dev servers")
    p_serve.add_argument("--api-only", action="store_true", help="Skip Next.js")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(dev_handler=cmd_serve)

    p_docker = dev_sub.add_parser("docker", help="docker compose up + health check")
    p_docker.add_argument("--profile", help="Compose profile (e.g. hosted)")
    p_docker.add_argument("--build", action="store_true")
    p_docker.add_argument("--no-health", action="store_true")
    p_docker.add_argument("--health-url", default="http://127.0.0.1:8000/api/v1/health")
    p_docker.add_argument("--timeout", type=int, default=90)
    p_docker.set_defaults(dev_handler=cmd_docker)

    p_bench = dev_sub.add_parser("benchmark", help="Run scripts/run_benchmark.py")
    p_bench.add_argument(
        "benchmark_args",
        nargs="*",
        help="Extra args forwarded to the benchmark script",
    )
    p_bench.set_defaults(dev_handler=cmd_benchmark)

    p_check = dev_sub.add_parser(
        "check",
        help="lint (python) + test — mirrors make check / CI core",
    )
    p_check.set_defaults(dev_handler=cmd_check)


def run_dev(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``dev`` subcommand."""
    handler = getattr(args, "dev_handler", None)
    if handler is None:
        print("error: no dev subcommand selected; try: nightmarenet dev --help", file=sys.stderr)
        return 2
    return int(handler(args))
