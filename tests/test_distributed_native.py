"""Tests for distributed training and checkpointing."""

import json
import os
from unittest import mock

import pytest
import torch
import torch.nn as nn

import nightmarenet
from nightmarenet.distributed.checkpoint import (
    AtomicCheckpointer,
    check_version_compatibility,
    compute_config_hash,
    validate_checkpoint_integrity,
)
from nightmarenet.distributed.ddp_wrapper import DDPWrapper
from nightmarenet.distributed.device_pool import DevicePool
from nightmarenet.distributed.resume import ResumeManager
from nightmarenet.distributed.strategies import apply_phase_strategy, unwrap_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


def test_device_pool():
    pool = DevicePool(override_devices=[0, 1, 2])
    assert pool.get_num_devices() == 3
    assert pool.should_use_ddp() is True

    pool2 = DevicePool(override_devices=[0])
    assert pool2.get_num_devices() == 1
    assert pool2.should_use_ddp() is False

    assert pool.estimate_memory_requirements(1000) > 0


def test_compute_config_hash():
    config = {"training": {"batch_size": 32}, "model": {"name": "test"}}
    hash1 = compute_config_hash(config)
    hash2 = compute_config_hash(config)

    config["training"]["batch_size"] = 16
    hash3 = compute_config_hash(config)

    assert hash1 == hash2
    assert hash1 != hash3


def test_atomic_checkpoint_and_resume(tmp_path):
    base_dir = tmp_path / "checkpoints"
    checkpointer = AtomicCheckpointer(str(base_dir))

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {"test": 123}

    target_dir = checkpointer.save(
        run_id="test_run",
        cycle=1,
        phase="wake",
        model=model,
        optimizer=optimizer,
        config=config,
        metrics={"loss": 0.5},
    )

    assert os.path.exists(target_dir)
    assert os.path.exists(os.path.join(target_dir, ".complete"))
    assert os.path.exists(os.path.join(target_dir, "metadata.json"))

    # Resume
    resume_mgr = ResumeManager(target_dir)
    new_model = SimpleModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

    metadata = resume_mgr.verify_and_load(new_model, new_optimizer, config)
    assert metadata["cycle"] == 1
    assert metadata["phase"] == "wake"


@mock.patch("nightmarenet.distributed.strategies.logger")
def test_apply_phase_strategy(mock_logger):
    model = SimpleModel()
    device_pool = DevicePool(override_devices=[0, 1])
    ddp_wrapper = DDPWrapper()
    ddp_wrapper.is_initialized = True  # Mock initialized

    # Wake phase
    with mock.patch.object(ddp_wrapper, "wrap_model", return_value="ddp_model"):
        res = apply_phase_strategy("wake", model, device_pool, ddp_wrapper)
        assert res == "ddp_model"

    # Dream phase (DataParallel)
    res = apply_phase_strategy("dream", model, device_pool, ddp_wrapper)
    assert isinstance(res, nn.DataParallel)

    # Compress phase (Single GPU)
    res = apply_phase_strategy("compress", model, device_pool, ddp_wrapper)
    assert res is model


# ============================================================================
# DDP Wrapper Tests
# ============================================================================


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
@mock.patch("nightmarenet.distributed.ddp_wrapper.torch.cuda")
def test_ddp_wrapper_setup_with_torchrun(mock_cuda, mock_dist, monkeypatch):
    """Test DDP initialization when launched via torchrun."""
    mock_dist.is_available.return_value = True
    mock_dist.is_initialized.return_value = False
    mock_dist.get_rank.return_value = 0
    mock_cuda.is_available.return_value = True

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    wrapper = DDPWrapper(backend="nccl")
    wrapper.setup()

    mock_dist.init_process_group.assert_called_once_with(backend="nccl")
    mock_cuda.set_device.assert_called_once_with(0)
    assert wrapper.is_initialized is True


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
def test_ddp_wrapper_setup_without_torchrun(mock_dist, monkeypatch):
    """Test DDP initialization when not launched via torchrun."""
    mock_dist.is_available.return_value = True
    mock_dist.is_initialized.return_value = False

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    wrapper = DDPWrapper()
    wrapper.setup()

    mock_dist.init_process_group.assert_not_called()
    assert wrapper.is_initialized is False


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
def test_ddp_wrapper_setup_dist_not_available(mock_dist):
    """Test DDP initialization when torch.distributed is not available."""
    mock_dist.is_available.return_value = False

    wrapper = DDPWrapper()
    wrapper.setup()

    mock_dist.init_process_group.assert_not_called()
    assert wrapper.is_initialized is False


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
@mock.patch("nightmarenet.distributed.ddp_wrapper.torch")
def test_ddp_wrapper_wrap_model(mock_torch, mock_dist, monkeypatch):
    """Test model wrapping with DDP."""
    mock_dist.is_initialized.return_value = True
    mock_torch.cuda.is_available.return_value = True

    model = SimpleModel()
    wrapper = DDPWrapper()
    wrapper.is_initialized = True

    monkeypatch.setenv("LOCAL_RANK", "0")
    with mock.patch.object(model, "to", return_value=model) as mock_to:
        with mock.patch("nightmarenet.distributed.ddp_wrapper.DistributedDataParallel") as mock_ddp:
            mock_ddp.return_value = "wrapped_model"
            result = wrapper.wrap_model(model)
            assert result == "wrapped_model"
            mock_to.assert_called_once_with(0)
            mock_ddp.assert_called_once_with(model, device_ids=[0])


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
@mock.patch("nightmarenet.distributed.ddp_wrapper.torch")
def test_ddp_wrapper_wrap_model_no_cuda(mock_torch, mock_dist):
    """Test model wrapping when CUDA is not available."""
    mock_dist.is_initialized.return_value = True
    mock_torch.cuda.is_available.return_value = False

    model = SimpleModel()
    wrapper = DDPWrapper()
    wrapper.is_initialized = True

    result = wrapper.wrap_model(model)
    assert result is model


@mock.patch("nightmarenet.distributed.ddp_wrapper.dist")
def test_ddp_wrapper_teardown(mock_dist):
    """Test DDP teardown."""
    mock_dist.is_initialized.return_value = True

    wrapper = DDPWrapper()
    wrapper.is_initialized = True
    wrapper.teardown()

    mock_dist.destroy_process_group.assert_called_once()
    assert wrapper.is_initialized is False


# ============================================================================
# Strategy Tests
# ============================================================================


def test_unwrap_model_simple():
    """Test unwrapping a simple model."""
    model = SimpleModel()
    result = unwrap_model(model)
    assert result is model


def test_unwrap_model_ddp():
    """Test unwrapping a DDP model."""
    model = SimpleModel()
    ddp_model = nn.DataParallel(model)
    result = unwrap_model(ddp_model)
    assert result is model


def test_unwrap_model_nested():
    """Test unwrapping a nested wrapped model."""
    model = SimpleModel()
    ddp_model = nn.DataParallel(model)
    nested = nn.DataParallel(ddp_model)
    result = unwrap_model(nested)
    assert result is model


@mock.patch("nightmarenet.distributed.strategies.logger")
def test_strategy_nightmare_phase(mock_logger):
    """Test nightmare phase strategy (should use DDP like wake)."""
    model = SimpleModel()
    device_pool = DevicePool(override_devices=[0, 1])
    ddp_wrapper = DDPWrapper()
    ddp_wrapper.is_initialized = True

    with mock.patch.object(ddp_wrapper, "wrap_model", return_value="ddp_model"):
        res = apply_phase_strategy("nightmare", model, device_pool, ddp_wrapper)
        assert res == "ddp_model"


@mock.patch("nightmarenet.distributed.strategies.logger")
def test_strategy_single_device_fallback(mock_logger):
    """Test strategy fallback when only one device is available."""
    model = SimpleModel()
    device_pool = DevicePool(override_devices=[0])
    ddp_wrapper = DDPWrapper()
    ddp_wrapper.is_initialized = True

    res = apply_phase_strategy("wake", model, device_pool, ddp_wrapper)
    assert res is model


@mock.patch("nightmarenet.distributed.strategies.logger")
def test_strategy_ddp_not_initialized_fallback(mock_logger):
    """Test strategy fallback when DDP is not initialized."""
    model = SimpleModel()
    device_pool = DevicePool(override_devices=[0, 1])
    ddp_wrapper = DDPWrapper()
    ddp_wrapper.is_initialized = False

    res = apply_phase_strategy("wake", model, device_pool, ddp_wrapper)
    assert res is model


# ============================================================================
# Checkpoint Edge Case Tests
# ============================================================================


def test_checkpoint_missing_directory(tmp_path):
    """Test loading from a non-existent checkpoint directory."""
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_checkpoint_integrity(str(tmp_path / "nonexistent"))


def test_checkpoint_missing_sentinel(tmp_path):
    """Test checkpoint without .complete sentinel file."""
    checkpoint_dir = tmp_path / "incomplete"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text("{}")

    with pytest.raises(ValueError, match="incomplete"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_missing_metadata(tmp_path):
    """Test checkpoint without metadata.json file."""
    checkpoint_dir = tmp_path / "no_metadata"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")

    with pytest.raises(ValueError, match="metadata missing"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_invalid_json(tmp_path):
    """Test checkpoint with corrupted metadata.json."""
    checkpoint_dir = tmp_path / "invalid_json"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")
    (checkpoint_dir / "metadata.json").write_text("{invalid json}")

    with pytest.raises(ValueError, match="Failed to parse"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_missing_required_keys(tmp_path):
    """Test checkpoint with incomplete metadata."""
    checkpoint_dir = tmp_path / "missing_keys"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")
    (checkpoint_dir / "metadata.json").write_text('{"version": "1.0.0"}')

    with pytest.raises(ValueError, match="missing required key"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_missing_model_weights(tmp_path):
    """Test checkpoint without any model weight files."""
    checkpoint_dir = tmp_path / "no_weights"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": nightmarenet.__version__,
                "cycle": 1,
                "phase": "wake",
                "config_hash": "abc",
            }
        )
    )

    with pytest.raises(ValueError, match="does not contain any valid model weights"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_missing_optimizer_state(tmp_path):
    """Test checkpoint without optimizer state file."""
    checkpoint_dir = tmp_path / "no_optimizer"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": nightmarenet.__version__,
                "cycle": 1,
                "phase": "wake",
                "config_hash": "abc",
            }
        )
    )
    (checkpoint_dir / "model.pt").write_text("dummy")

    with pytest.raises(ValueError, match="missing required state file"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_checksum_mismatch(tmp_path):
    """Test checkpoint with file checksum mismatch."""
    checkpoint_dir = tmp_path / "checksum_mismatch"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".complete").write_text("complete")

    # Create a model file
    model_file = checkpoint_dir / "model.pt"
    model_file.write_text("original content")

    # Create metadata with wrong checksum
    metadata = {
        "version": nightmarenet.__version__,
        "cycle": 1,
        "phase": "wake",
        "config_hash": "abc",
        "file_hashes": {"model.pt": "wrong_hash"},
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))
    (checkpoint_dir / "optimizer.pt").write_text("dummy")
    (checkpoint_dir / "rng_state.pt").write_text("dummy")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_checkpoint_integrity(str(checkpoint_dir))


def test_checkpoint_save_overwrites_existing(tmp_path):
    """Test that saving a checkpoint overwrites an existing one."""
    base_dir = tmp_path / "checkpoints"
    checkpointer = AtomicCheckpointer(str(base_dir))

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {"test": 123}

    # First save
    target_dir1 = checkpointer.save(
        run_id="test_run",
        cycle=1,
        phase="wake",
        model=model,
        optimizer=optimizer,
        config=config,
        metrics={"loss": 0.5},
    )

    # Second save (should overwrite)
    target_dir2 = checkpointer.save(
        run_id="test_run",
        cycle=1,
        phase="wake",
        model=model,
        optimizer=optimizer,
        config=config,
        metrics={"loss": 0.3},
    )

    assert target_dir1 == target_dir2
    assert os.path.exists(target_dir2)

    metadata_path = os.path.join(target_dir2, "metadata.json")
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["metrics"]["loss"] == 0.3


# ============================================================================
# Version Compatibility Tests
# ============================================================================


def test_version_compatibility_same():
    """Test version compatibility with identical versions."""
    check_version_compatibility("1.0.0", "1.0.0")


def test_version_compatibility_minor_diff():
    """Test version compatibility with different minor versions (same major)."""
    check_version_compatibility("1.2.0", "1.3.0")


def test_version_compatibility_major_mismatch():
    """Test version compatibility with major version mismatch."""
    with pytest.raises(ValueError, match="Major version mismatch"):
        check_version_compatibility("1.0.0", "2.0.0")


def test_version_compatibility_zero_minor_mismatch():
    """Test version compatibility with 0.x minor version mismatch."""
    with pytest.raises(ValueError, match="Minor version mismatch in 0.x release"):
        check_version_compatibility("0.1.0", "0.2.0")


def test_version_compatibility_invalid_format():
    """Test version compatibility with invalid version format."""
    with pytest.raises(ValueError, match="Invalid version format"):
        check_version_compatibility("invalid", "1.0.0")


# ============================================================================
# Config Hash Determinism Tests
# ============================================================================


def test_config_hash_determinism():
    """Test that config hash is deterministic across calls."""
    config = {"training": {"batch_size": 32, "lr": 0.001}, "model": {"layers": 3}}

    hash1 = compute_config_hash(config)
    hash2 = compute_config_hash(config)
    hash3 = compute_config_hash(config)

    assert hash1 == hash2 == hash3


def test_config_hash_order_independence():
    """Test that config hash is independent of key order."""
    config1 = {"a": 1, "b": 2, "c": 3}
    config2 = {"c": 3, "a": 1, "b": 2}

    hash1 = compute_config_hash(config1)
    hash2 = compute_config_hash(config2)

    assert hash1 == hash2


def test_config_hash_nested_order_independence():
    """Test that config hash is independent of nested key order."""
    config1 = {"outer": {"inner": {"x": 1, "y": 2}}}
    config2 = {"outer": {"inner": {"y": 2, "x": 1}}}

    hash1 = compute_config_hash(config1)
    hash2 = compute_config_hash(config2)

    assert hash1 == hash2


# ============================================================================
# Device Pool Tests
# ============================================================================


def test_device_pool_override():
    """Test device pool with explicit device override."""
    pool = DevicePool(override_devices=[0, 2, 4])
    assert pool.available_devices == [0, 2, 4]
    assert pool.get_num_devices() == 3


def test_device_pool_empty_override():
    """Test device pool with empty override."""
    pool = DevicePool(override_devices=[])
    assert pool.available_devices == []
    assert pool.get_num_devices() == 0


def test_device_pool_no_cuda():
    """Test device pool when CUDA is not available."""
    with mock.patch(
        "nightmarenet.distributed.device_pool.torch.cuda.is_available",
        return_value=False,
    ):
        pool = DevicePool()
        assert pool.available_devices == []
        assert pool.get_num_devices() == 0


def test_device_pool_memory_estimation():
    """Test memory requirement estimation."""
    pool = DevicePool()

    # 1M parameters
    mem_1m = pool.estimate_memory_requirements(1_000_000)
    # 10M parameters
    mem_10m = pool.estimate_memory_requirements(10_000_000)

    assert mem_10m > mem_1m
    assert mem_10m == mem_1m * 10


def test_device_pool_should_use_ddp():
    """Test DDP feasibility check."""
    pool_multi = DevicePool(override_devices=[0, 1])
    assert pool_multi.should_use_ddp() is True

    pool_single = DevicePool(override_devices=[0])
    assert pool_single.should_use_ddp() is False

    pool_empty = DevicePool(override_devices=[])
    assert pool_empty.should_use_ddp() is False
