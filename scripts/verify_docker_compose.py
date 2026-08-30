#!/usr/bin/env python3
"""
verify_docker_compose.py
-------------------------
Sequential health verification for the NightmareNet Docker Compose stack.

Required services (see ``docker-compose.yml``, the source of truth for
service names, ports, and health checks):

    api       — FastAPI backend. Always checked; started by a bare
                ``docker compose up`` (no profile required).
    redis     — Message broker / cache. Part of the ``hosted`` profile
                (``docker compose --profile hosted up``).
    db        — PostgreSQL. Part of the ``hosted`` profile.
    worker    — Celery worker. Part of the ``hosted`` profile.

As of this writing, ``api`` does not connect to ``redis``/``db`` at
runtime (see the comments at the top of ``docker-compose.yml``) — that
wiring belongs to the planned ``nightmarenet_server/`` hosted platform.
Because of that, this script auto-detects which services are actually
running: services that are part of the ``hosted`` profile and are not up
are reported as SKIPPED (not FAILED) unless ``--profile hosted`` is passed
explicitly, in which case a missing hosted service is a hard failure. The
API check always runs and is always required.

Usage
-----
    python scripts/verify_docker_compose.py
    python scripts/verify_docker_compose.py --profile hosted
    make verify-stack

Environment
-----------
Reads the same environment variables/defaults as ``docker-compose.yml``:

    API_PORT (default 8000), POSTGRES_PORT (default 5432),
    REDIS_PORT (default 6379), POSTGRES_USER (default nightmare),
    POSTGRES_DB (default nightmarenet), COMPOSE_HOST (default 127.0.0.1)

Exit codes
----------
0 — every required check passed (skipped optional services do not count
    as failures).
1 — at least one required check failed or timed out. A missing
    ``docker compose`` / ``docker-compose`` CLI is only a hard failure
    under ``--profile hosted`` (the compose-dependent checks are skipped
    otherwise, since only the API is required by default).

Dependencies
------------
Standard library plus ``requests`` (already a core project dependency —
see ``pyproject.toml``). No other third-party packages are used.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import requests

# (ok, message, skipped)
CheckOutcome = Tuple[bool, str, bool]

# Mirrors the `test:` commands defined in docker-compose.yml / Dockerfiles.
# Keep these in sync with that file if service healthchecks change.
_REDIS_PING_CMD = ["redis-cli", "ping"]
_WORKER_HEALTHCHECK_CMD = ["python", "/usr/local/bin/healthcheck_worker.py"]

Status = str  # one of "PASS", "FAIL", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    duration_s: float


def _find_compose_command() -> Optional[List[str]]:
    """Return an argv prefix for a working Compose CLI, or None."""
    if shutil.which("docker"):
        probe = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            return ["docker", "compose"]
    if shutil.which("docker-compose"):
        probe = subprocess.run(
            ["docker-compose", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            return ["docker-compose"]
    return None


def _compose_exec(
    compose_cmd: Sequence[str],
    service: str,
    cmd: Sequence[str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    full = [*compose_cmd, "exec", "-T", service, *cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def _service_not_running(proc: subprocess.CompletedProcess[str]) -> bool:
    """Heuristic: does this failure mean "service isn't up" vs. a real error?"""
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    return any(
        phrase in text
        for phrase in (
            "is not running",
            "no such service",
            "no container found",
            "service .* is not running",
        )
    )


def run_check(
    name: str,
    fn: Callable[[], CheckOutcome],
    *,
    optional: bool,
    timeout: float,
) -> CheckResult:
    start = time.monotonic()
    try:
        ok, message, skipped = fn()
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return CheckResult(name, "FAIL", f"timed out after {timeout:.0f}s", elapsed)
    except Exception as exc:  # defensive: never let one check crash the run
        elapsed = time.monotonic() - start
        return CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}", elapsed)
    elapsed = time.monotonic() - start

    if skipped:
        return CheckResult(name, "SKIP", message, elapsed)
    if ok:
        return CheckResult(name, "PASS", message, elapsed)
    return CheckResult(name, "FAIL", message, elapsed)


def check_api_health(host: str, port: int, timeout: float, retries: int) -> CheckOutcome:
    url = f"http://{host}:{port}/api/v1/health"
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                try:
                    body = resp.json()
                except ValueError:
                    return False, f"non-JSON response body: {resp.text[:200]}", False
                if body.get("status") != "ok":
                    return False, f"unexpected body: {body}", False
                return True, f"200 OK, version={body.get('version')} ({url})", False
        if attempt < retries:
            time.sleep(min(2.0, timeout))
    return False, f"unreachable after {retries} attempts: {last_error} ({url})", False


def check_redis(
    compose_cmd: Sequence[str], service: str, timeout: float, hosted: bool
) -> CheckOutcome:
    try:
        proc = _compose_exec(compose_cmd, service, _REDIS_PING_CMD, timeout)
    except FileNotFoundError:
        return False, "docker compose executable not found", False
    if proc.returncode != 0:
        if not hosted and _service_not_running(proc):
            return False, f"{service} is not running (requires --profile hosted)", True
        return False, (proc.stderr or proc.stdout).strip() or "redis-cli ping failed", False
    reply = proc.stdout.strip()
    if reply != "PONG":
        return False, f"unexpected reply: {reply!r}", False
    return True, "PING -> PONG", False


def check_postgres(
    compose_cmd: Sequence[str],
    service: str,
    user: str,
    db: str,
    timeout: float,
    hosted: bool,
) -> CheckOutcome:
    cmd = ["pg_isready", "-U", user, "-d", db]
    try:
        proc = _compose_exec(compose_cmd, service, cmd, timeout)
    except FileNotFoundError:
        return False, "docker compose executable not found", False
    if proc.returncode != 0:
        if not hosted and _service_not_running(proc):
            return False, f"{service} is not running (requires --profile hosted)", True
        return False, (proc.stdout or proc.stderr).strip() or "pg_isready failed", False
    return True, proc.stdout.strip() or "accepting connections", False


def check_worker(
    compose_cmd: Sequence[str], service: str, timeout: float, hosted: bool
) -> CheckOutcome:
    try:
        proc = _compose_exec(compose_cmd, service, _WORKER_HEALTHCHECK_CMD, timeout)
    except FileNotFoundError:
        return False, "docker compose executable not found", False
    if proc.returncode != 0:
        if not hosted and _service_not_running(proc):
            return False, f"{service} is not running (requires --profile hosted)", True
        return False, (proc.stderr or proc.stdout).strip() or "healthcheck_worker.py failed", False
    return True, proc.stdout.strip() or "healthy", False


def check_container_reachability(
    compose_cmd: Sequence[str],
    from_service: str,
    target_host: str,
    target_port: int,
    timeout: float,
    hosted: bool,
) -> CheckOutcome:
    """TCP-level check that ``from_service``'s container can reach
    ``target_host:target_port`` over the shared ``nightmarenet`` network.

    This verifies network wiring only — it does not imply the application
    itself uses the target service today (the OSS `api` does not yet talk
    to Redis/Postgres; see the module docstring).
    """
    snippet = (
        "import socket,sys;"
        f"s=socket.create_connection(('{target_host}',{target_port}),timeout={timeout});"
        "s.close()"
    )
    cmd = ["python3", "-c", snippet]
    try:
        proc = _compose_exec(compose_cmd, from_service, cmd, timeout + 2)
    except FileNotFoundError:
        return False, "docker compose executable not found", False
    if proc.returncode != 0:
        combined = f"{proc.stdout}\n{proc.stderr}"
        if not hosted and (
            _service_not_running(proc) or "name or service not known" in combined.lower()
        ):
            return (
                False,
                f"{from_service} -> {target_host}:{target_port} unreachable "
                "(requires --profile hosted)",
                True,
            )
        return False, combined.strip() or "connection failed", False
    return True, f"{from_service} -> {target_host}:{target_port} reachable", False


def print_result(result: CheckResult) -> None:
    icon = {"PASS": "\u2713", "FAIL": "\u2717", "SKIP": "\u25cb"}[result.status]
    print(
        f"  {icon} [{result.status:<4}] {result.name:<32} "
        f"({result.duration_s:.2f}s) {result.message}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--profile",
        choices=["default", "hosted"],
        default="default",
        help=(
            "'default' expects only api+frontend and treats a down "
            "redis/db/worker as SKIP; 'hosted' requires every service and "
            "treats a down one as FAIL."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to reach the API on.")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--postgres-user", default="nightmare")
    parser.add_argument("--postgres-db", default="nightmarenet")
    parser.add_argument(
        "--api-retries",
        type=int,
        default=8,
        help="Retries for the API health check (handles container start_period).",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-check timeout in seconds.")
    args = parser.parse_args(argv)

    hosted = args.profile == "hosted"

    print(f"NightmareNet Docker Compose verification (profile={args.profile})\n")

    compose_cmd = _find_compose_command()
    results: List[CheckResult] = []

    print("Running checks:")

    if compose_cmd is None:
        # redis/db/worker/reachability checks all need `docker compose exec`;
        # without it they simply can't run. Under the default profile that's
        # a SKIP (the API-only stack doesn't need Compose to be inspectable);
        # under --profile hosted it's a hard requirement.
        results.append(
            CheckResult(
                "docker compose CLI",
                "FAIL" if hosted else "SKIP",
                "neither 'docker compose' nor 'docker-compose' is available",
                0.0,
            )
        )
        print_result(results[-1])
    else:
        print(f"  (using compose command: {' '.join(compose_cmd)})")

    results.append(
        run_check(
            "api: GET /api/v1/health",
            lambda: check_api_health(args.host, args.api_port, args.timeout, args.api_retries),
            optional=False,
            timeout=args.timeout,
        )
    )
    print_result(results[-1])

    if compose_cmd is not None:
        results.append(
            run_check(
                "redis: PING",
                lambda: check_redis(compose_cmd, "redis", args.timeout, hosted),
                optional=not hosted,
                timeout=args.timeout,
            )
        )
        print_result(results[-1])

        results.append(
            run_check(
                "postgres: pg_isready",
                lambda: check_postgres(
                    compose_cmd, "db", args.postgres_user, args.postgres_db, args.timeout, hosted
                ),
                optional=not hosted,
                timeout=args.timeout,
            )
        )
        print_result(results[-1])

        results.append(
            run_check(
                "worker: healthcheck_worker.py",
                lambda: check_worker(compose_cmd, "worker", args.timeout, hosted),
                optional=not hosted,
                timeout=args.timeout,
            )
        )
        print_result(results[-1])

        results.append(
            run_check(
                "api -> redis: TCP reachability",
                lambda: check_container_reachability(
                    compose_cmd, "api", "redis", 6379, args.timeout, hosted
                ),
                optional=not hosted,
                timeout=args.timeout,
            )
        )
        print_result(results[-1])

        results.append(
            run_check(
                "api -> postgres: TCP reachability",
                lambda: check_container_reachability(
                    compose_cmd, "api", "db", 5432, args.timeout, hosted
                ),
                optional=not hosted,
                timeout=args.timeout,
            )
        )
        print_result(results[-1])

    total_s = sum(r.duration_s for r in results)
    passed = sum(1 for r in results if r.status == "PASS")
    skipped = sum(1 for r in results if r.status == "SKIP")
    failed = sum(1 for r in results if r.status == "FAIL")

    print(
        f"\n{passed} passed, {skipped} skipped, {failed} failed "
        f"in {total_s:.2f}s (profile={args.profile})"
    )

    if failed:
        print("\nFAIL: one or more required checks did not pass.", file=sys.stderr)
        return 1

    print("\nPASS: stack is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
