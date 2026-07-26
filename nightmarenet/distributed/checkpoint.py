"""Atomic checkpointing for distributed execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def compute_config_hash(config: dict) -> str:
    """Compute a deterministic hash of the training configuration."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


class AtomicCheckpointer:
    """Handles atomic saves of model, optimizer, and phase state."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(
        self,
        run_id: str,
        cycle: int,
        phase: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        metrics: Optional[dict] = None,
        devices_used: Optional[list[int]] = None,
    ) -> str:
        """Save state atomically and drop a .complete sentinel."""
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        target_dir = os.path.join(run_dir, f"cycle-{cycle}-{phase}")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Write to a temporary directory first
        temp_dir = tempfile.mkdtemp(dir=run_dir, prefix=f".tmp_cycle-{cycle}-{phase}_")
        try:
            # 1. Model weights
            model_path = os.path.join(temp_dir, "model.pt")

            # Handle DDP / DataParallel unwrapping
            model_to_save = model.module if hasattr(model, "module") else model
            if hasattr(model_to_save, "save_pretrained"):
                model_to_save.save_pretrained(temp_dir)
            else:
                torch.save(model_to_save.state_dict(), model_path)

            # 2. Optimizer state
            opt_path = os.path.join(temp_dir, "optimizer.pt")
            torch.save(optimizer.state_dict(), opt_path)

            # 3. RNG States
            rng_path = os.path.join(temp_dir, "rng_state.pt")
            torch.save(
                {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                rng_path,
            )

            # 4. Metadata and Config hash
            import time

            from nightmarenet import __version__ as app_version

            meta_path = os.path.join(temp_dir, "metadata.json")
            file_hashes = compute_dir_hashes(temp_dir)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "version": app_version,
                        "timestamp": time.time(),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "cycle": cycle,
                        "phase": phase,
                        "config_hash": compute_config_hash(config),
                        "metrics": metrics or {},
                        "devices_used": devices_used or [],
                        "file_hashes": file_hashes,
                    },
                    f,
                    indent=2,
                )

            # Atomic save: sentinel inside temp, then swap with backup
            sentinel_path = os.path.join(temp_dir, ".complete")
            with open(sentinel_path, "w") as f:
                f.write("complete")

            if os.path.exists(target_dir):
                backup_dir = target_dir + ".bak"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                os.rename(target_dir, backup_dir)
                try:
                    os.rename(temp_dir, target_dir)
                    shutil.rmtree(backup_dir, ignore_errors=True)
                except Exception:
                    os.rename(backup_dir, target_dir)
                    raise
            else:
                os.rename(temp_dir, target_dir)

            logger.info(f"Atomically saved checkpoint to {target_dir}")
            return target_dir

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.error(f"Failed to save checkpoint: {e}")
            raise


def load_model_weights(model: torch.nn.Module, checkpoint_dir: str, device: torch.device) -> None:
    """Load model weights from a checkpoint directory into an existing model."""
    model_to_load = model.module if hasattr(model, "module") else model

    # Try model.pt first
    model_pt = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_pt):
        logger.info("Loading model state dict from %s", model_pt)
        state_dict = torch.load(model_pt, map_location=device)
        model_to_load.load_state_dict(state_dict, strict=False)
        return

    # Try safetensors
    model_safetensors = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(model_safetensors):
        logger.info("Loading model safetensors from %s", model_safetensors)
        try:
            from safetensors.torch import load_file

            state_dict = load_file(model_safetensors, device=str(device))
            model_to_load.load_state_dict(state_dict, strict=False)
            return
        except ImportError:
            logger.warning("safetensors is not installed but model.safetensors exists.")

    # Try pytorch_model.bin
    model_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_bin):
        logger.info("Loading model weights from %s", model_bin)
        state_dict = torch.load(model_bin, map_location=device)
        model_to_load.load_state_dict(state_dict, strict=False)
        return

    logger.warning("No model weight file found in %s", checkpoint_dir)


def compute_file_sha256(filepath: str) -> str:
    """Compute the SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dir_hashes(directory: str) -> dict[str, str]:
    """Compute SHA256 hashes for all files in a directory, relative to directory."""
    hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file in ("metadata.json", ".complete"):
                continue
            filepath = os.path.join(root, file)
            relpath = os.path.relpath(filepath, directory)
            # Use forward slashes for OS portability in json
            relpath_key = relpath.replace("\\", "/")
            hashes[relpath_key] = compute_file_sha256(filepath)
    return hashes


def check_version_compatibility(checkpoint_version: str, current_version: str) -> None:
    """Check if checkpoint_version is compatible with current_version."""
    try:
        chk_parts = [int(x) for x in checkpoint_version.split(".")]
        cur_parts = [int(x) for x in current_version.split(".")]
    except ValueError as e:
        raise ValueError(
            f"Invalid version format (checkpoint: {checkpoint_version}, current: {current_version})"
        ) from e

    # If major version differs, or if major version is 0 and minor version differs,
    # they are incompatible.
    if chk_parts[0] != cur_parts[0]:
        raise ValueError(
            f"Incompatible checkpoint version {checkpoint_version} with "
            f"current version {current_version}. Major version mismatch."
        )
    if chk_parts[0] == 0 and chk_parts[1] != cur_parts[1]:
        raise ValueError(
            f"Incompatible checkpoint version {checkpoint_version} with "
            f"current version {current_version}. Minor version mismatch in 0.x release."
        )


def validate_checkpoint_integrity(checkpoint_dir: str, config: Optional[dict] = None) -> dict:
    """Validate the integrity and structure of a checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found.")

    sentinel_path = os.path.join(checkpoint_dir, ".complete")
    if not os.path.exists(sentinel_path):
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} is incomplete (.complete sentinel missing)."
        )

    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"Checkpoint metadata missing in {checkpoint_dir}.")

    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse metadata.json in {checkpoint_dir}: {e}") from e

    # 1. Verify structure - check that required keys are in metadata
    required_keys = ["version", "cycle", "phase", "config_hash"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Checkpoint metadata is missing required key: {key}")

    # 2. Check version compatibility
    from nightmarenet import __version__ as app_version

    check_version_compatibility(metadata["version"], app_version)

    # 3. Check config hash if current config is provided
    if config is not None:
        expected_hash = compute_config_hash(config)
        if metadata["config_hash"] != expected_hash:
            logger.warning(
                "Config hash mismatch on resume. Training configuration may have changed."
            )

    # 4. Check that core files exist
    # A valid checkpoint must have at least one weight file and one optimizer state file
    weight_files = ["model.pt", "pytorch_model.bin", "model.safetensors"]
    has_weights = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in weight_files)
    if not has_weights:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} does not contain any valid model weights."
        )

    required_files = ["optimizer.pt", "rng_state.pt"]
    for req_file in required_files:
        if not os.path.exists(os.path.join(checkpoint_dir, req_file)):
            raise ValueError(
                f"Checkpoint at {checkpoint_dir} is missing required state file: {req_file}"
            )

    # 5. Checksum/hash validation
    recorded_hashes = metadata.get("file_hashes", {})
    if recorded_hashes:
        for relpath, recorded_hash in recorded_hashes.items():
            filepath = os.path.join(checkpoint_dir, relpath)
            if not os.path.exists(filepath):
                raise ValueError(f"Checkpoint file recorded in metadata is missing: {relpath}")
            current_hash = compute_file_sha256(filepath)
            if current_hash != recorded_hash:
                raise ValueError(f"Integrity check failed: Checksum mismatch for file '{relpath}'.")
        logger.info("Checksum validation passed successfully.")
    else:
        logger.warning("No file checksums found in checkpoint metadata. Skipping integrity check.")

    return metadata
