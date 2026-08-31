"""Tests for the unified ``nightmarenet dev`` CLI (issue #701)."""

from __future__ import annotations

import argparse

import pytest

from nightmarenet import cli, dev_cli


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def test_dev_help_lists_commands(capsys):
    parser = cli.build_parser()
    assert "dev" in parser.format_help()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["dev", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("lint", "test", "format", "migrate", "serve", "docker", "benchmark", "check"):
        assert name in out


def test_dev_subcommands_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["dev", "lint", "--python-only"])
    assert args.command == "dev"
    assert args.dev_command == "lint"
    assert args.python_only is True
    assert callable(args.dev_handler)


def test_missing_tool_message(monkeypatch, capsys):
    monkeypatch.setattr(dev_cli, "_which", lambda _name: None)
    code = dev_cli.cmd_lint(_ns(python_only=True))
    assert code == 127
    err = capsys.readouterr().err
    assert "ruff not found" in err
    assert "pip install" in err


def test_test_frontend_requires_npm(monkeypatch, capsys):
    monkeypatch.setattr(dev_cli, "_which", lambda name: None if name == "npm" else "/bin/true")
    code = dev_cli.cmd_test(_ns(frontend=True, marker=None, pytest_args=None))
    assert code == 127
    assert "Node.js not found" in capsys.readouterr().err


def test_run_invokes_subprocess(monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, env=None, check=False):  # noqa: ARG001
        calls.append(list(cmd))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(dev_cli.subprocess, "run", fake_run)
    code = dev_cli._run(["echo", "hi"])
    assert code == 0
    assert calls == [["echo", "hi"]]


def test_migrate_missing_alembic(monkeypatch, capsys):
    monkeypatch.setattr(dev_cli, "_which", lambda _n: None)
    code = dev_cli.cmd_migrate(_ns(revision="head"))
    assert code == 127
    assert "alembic not found" in capsys.readouterr().err


def test_main_dispatches_dev(monkeypatch):
    called = {}

    def fake_run_dev(args):
        called["ok"] = args.dev_command
        return 0

    monkeypatch.setattr("nightmarenet.dev_cli.run_dev", fake_run_dev)
    code = cli.main(["dev", "check"])
    assert code == 0
    assert called.get("ok") == "check"


def test_docker_missing_docker(monkeypatch, capsys):
    monkeypatch.setattr(dev_cli, "_which", lambda _n: None)
    code = dev_cli.cmd_docker(
        _ns(profile=None, build=False, no_health=True, health_url="", timeout=1)
    )
    assert code == 127
    assert "docker not found" in capsys.readouterr().err
