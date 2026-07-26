"""Tests for the `nightmarenet --version` CLI flag and verbosity flags."""

import pytest

from nightmarenet import __version__
from nightmarenet.cli import build_parser


def test_version_flag_exits_zero(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0


def test_version_flag_prints_installed_version(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])

    captured = capsys.readouterr()
    assert __version__ in captured.out
    assert "nightmarenet" in captured.out


@pytest.mark.parametrize(
    "flag,expected_verbose,expected_quiet",
    [
        ("--verbose", True, False),
        ("-v", True, False),
        ("--quiet", False, True),
        ("-q", False, True),
    ],
)
def test_verbosity_flags_parse(flag, expected_verbose, expected_quiet):
    parser = build_parser()
    args = parser.parse_args([flag, "train", "--config", "test.yaml"])
    assert args.verbose is expected_verbose
    assert args.quiet is expected_quiet


def test_default_verbosity():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "test.yaml"])
    assert args.verbose is False
    assert args.quiet is False


def test_verbose_and_quiet_mutually_exclusive(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--verbose", "--quiet", "train", "--config", "test.yaml"])

    captured = capsys.readouterr()
    assert "not allowed with argument" in captured.err
