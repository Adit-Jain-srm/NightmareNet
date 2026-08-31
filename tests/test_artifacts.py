import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from nightmarenet.artifacts.manager import ArtifactManager, parse_duration
from nightmarenet.cli import build_parser, cmd_artifacts


@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture providing a temporary directory as the workspace root."""
    return tmp_path


def test_parse_duration():
    assert parse_duration("30d") == 30 * 24 * 3600
    assert parse_duration("12h") == 12 * 3600
    assert parse_duration("45m") == 45 * 60
    assert parse_duration("60s") == 60.0
    assert parse_duration("100") == 100.0
    with pytest.raises(ValueError):
        parse_duration("invalid")


def test_register_file_and_directory(temp_workspace):
    manager = ArtifactManager(root_dir=temp_workspace)

    # 1. Register a file artifact
    file_path = temp_workspace / "test_log.log"
    file_path.write_text("Hello log content")

    meta_path = manager.register(
        path=file_path,
        run_id="run-1",
        artifact_type="log",
        retention_policy={"type": "TIME_BASED", "duration": "10d"},
    )

    assert meta_path.exists()
    assert meta_path.name == "test_log.log.artifact-meta.json"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["artifact_path"] == "test_log.log"
    assert meta["run_id"] == "run-1"
    assert meta["artifact_type"] == "log"
    assert meta["size_bytes"] == len("Hello log content")
    assert meta["retention_policy"] == {"type": "TIME_BASED", "duration": "10d"}
    assert "creation_time" in meta

    # 2. Register a directory artifact
    dir_path = temp_workspace / "checkpoint_epoch_1"
    dir_path.mkdir()
    (dir_path / "model.bin").write_text("weights_data")
    (dir_path / "config.json").write_text("config_data")

    meta_dir_path = manager.register(
        path=dir_path,
        run_id="run-1",
        artifact_type="checkpoint",
        retention_policy={"type": "KEEP_FOREVER"},
    )

    assert meta_dir_path.exists()
    with open(meta_dir_path, encoding="utf-8") as f:
        meta_dir = json.load(f)

    assert meta_dir["artifact_path"] == "checkpoint_epoch_1"
    # Total directory size is sum of sizes of files inside
    assert meta_dir["size_bytes"] == len("weights_data") + len("config_data")
    assert meta_dir["retention_policy"] == {"type": "KEEP_FOREVER"}


def test_list_artifacts(temp_workspace):
    manager = ArtifactManager(root_dir=temp_workspace)

    # Register two separate artifacts
    f1 = temp_workspace / "f1.txt"
    f1.write_text("f1")
    manager.register(f1, "run-10", "log")

    f2 = temp_workspace / "f2.txt"
    f2.write_text("f2")
    manager.register(f2, "run-20", "checkpoint")

    artifacts = manager.list_artifacts()
    assert len(artifacts) == 2

    # Filter by run_id
    run_10_arts = manager.list_artifacts(run_id="run-10")
    assert len(run_10_arts) == 1
    assert run_10_arts[0]["artifact_path"] == "f1.txt"


def test_clean_time_based_policy(temp_workspace):
    manager = ArtifactManager(root_dir=temp_workspace)

    # 1. New artifact (within 10 days)
    f_new = temp_workspace / "new.log"
    f_new.write_text("new")
    meta_new_path = manager.register(
        f_new,
        "run-1",
        "log",
        retention_policy={"type": "TIME_BASED", "duration": "10d"},
    )

    # 2. Old artifact (older than 10 days)
    f_old = temp_workspace / "old.log"
    f_old.write_text("old")
    meta_old_path = manager.register(
        f_old,
        "run-1",
        "log",
        retention_policy={"type": "TIME_BASED", "duration": "10d"},
    )

    # Manually tweak creation_time in sidecar metadata to be 15 days ago
    with open(meta_old_path, encoding="utf-8") as f:
        meta_old = json.load(f)
    fifteen_days_ago = datetime.now(timezone.utc) - timedelta(days=15)
    meta_old["creation_time"] = fifteen_days_ago.isoformat() + "Z"
    with open(meta_old_path, "w", encoding="utf-8") as f:
        json.dump(meta_old, f)

    # Clean
    deleted = manager.clean()

    # The clean function returns target paths that were deleted.
    # So f_old should be in deleted, and f_new should not.
    assert f_old in deleted
    assert f_new not in deleted
    assert not f_old.exists()
    assert not meta_old_path.exists()

    assert f_new.exists()
    assert meta_new_path.exists()


def test_clean_count_based_policy(temp_workspace):
    manager = ArtifactManager(root_dir=temp_workspace)

    # Register 4 checkpoints, with policy to keep last 2
    paths = []
    meta_paths = []
    policy = {"type": "COUNT_BASED", "count": 2}

    for i in range(4):
        p = temp_workspace / f"epoch_{i}"
        p.mkdir()
        (p / "model.pt").write_text("weights")
        paths.append(p)
        meta_p = manager.register(p, "run-1", "checkpoint", retention_policy=policy)
        meta_paths.append(meta_p)

        # Space out creation times slightly so we can sort reliably
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        creation = datetime.now(timezone.utc) - timedelta(minutes=(10 - i))
        meta["creation_time"] = creation.isoformat() + "Z"
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    # Clean
    deleted = manager.clean()

    assert len(deleted) == 2
    # The oldest two (epoch_0 and epoch_1) should be cleaned
    assert not paths[0].exists()
    assert not meta_paths[0].exists()
    assert not paths[1].exists()
    assert not meta_paths[1].exists()

    # The newest two (epoch_2 and epoch_3) should be retained
    assert paths[2].exists()
    assert meta_paths[2].exists()
    assert paths[3].exists()
    assert meta_paths[3].exists()


def test_clean_keep_forever_and_global_override(temp_workspace):
    manager = ArtifactManager(root_dir=temp_workspace)

    # 1. KEEP_FOREVER artifact
    f_forever = temp_workspace / "model.pt"
    f_forever.write_text("forever")
    meta_forever = manager.register(
        f_forever, "run-1", "checkpoint", retention_policy={"type": "KEEP_FOREVER"}
    )

    # Tweak creation time to be 40 days ago
    with open(meta_forever, encoding="utf-8") as f:
        meta = json.load(f)
    forty_days_ago = datetime.now(timezone.utc) - timedelta(days=40)
    meta["creation_time"] = forty_days_ago.isoformat() + "Z"
    with open(meta_forever, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Normal clean does not touch it
    deleted = manager.clean()
    assert len(deleted) == 0
    assert f_forever.exists()

    # Global override `--older-than 30d` cleans it
    deleted = manager.clean(older_than="30d")
    assert len(deleted) == 1
    assert not f_forever.exists()
    assert not meta_forever.exists()


def test_cli_subcommands(temp_workspace, capsys):
    manager = ArtifactManager(root_dir=temp_workspace)

    # Register an artifact
    f = temp_workspace / "cli_test.log"
    f.write_text("cli content")
    manager.register(f, "run-cli", "log", retention_policy={"type": "KEEP_FOREVER"})

    parser = build_parser()

    # 1. Test 'show' subcommand
    args_show = parser.parse_args(["artifacts", "show", str(f)])
    # Mock root_dir in ArtifactManager so the CLI cmd uses our test temp_workspace
    with mock.patch("nightmarenet.artifacts.manager.ArtifactManager") as mock_am_class:
        mock_am_instance = mock.MagicMock()
        mock_am_instance.root_dir = temp_workspace
        mock_am_class.return_value = mock_am_instance

        # Call show
        status = cmd_artifacts(args_show)
        assert status == 0

        captured = capsys.readouterr()
        # Verify JSON metadata printed to stdout
        data = json.loads(captured.out)
        assert data["artifact_path"] == "cli_test.log"
        assert data["run_id"] == "run-cli"

    # 2. Test 'list' subcommand
    args_list = parser.parse_args(["artifacts", "list"])
    with mock.patch("nightmarenet.artifacts.manager.ArtifactManager") as mock_am_class:
        mock_am_instance = mock.MagicMock()
        mock_am_instance.root_dir = temp_workspace
        mock_am_instance.list_artifacts.return_value = [
            {
                "artifact_path": "cli_test.log",
                "artifact_type": "log",
                "run_id": "run-cli",
                "size_bytes": 11,
                "creation_time": "2026-08-08T10:00:00Z",
            }
        ]
        mock_am_class.return_value = mock_am_instance

        status = cmd_artifacts(args_list)
        assert status == 0
        captured = capsys.readouterr()
        assert "cli_test.log" in captured.out
        assert "run-cli" in captured.out

    # 3. Test 'clean' subcommand
    args_clean = parser.parse_args(["artifacts", "clean", "--older-than", "30d"])
    with mock.patch("nightmarenet.artifacts.manager.ArtifactManager") as mock_am_class:
        mock_am_instance = mock.MagicMock()
        mock_am_instance.root_dir = temp_workspace
        mock_am_instance.clean.return_value = [Path("cli_test.log")]
        mock_am_class.return_value = mock_am_instance

        status = cmd_artifacts(args_clean)
        assert status == 0
        captured = capsys.readouterr()
        assert "Successfully cleaned 1 artifacts." in captured.out
