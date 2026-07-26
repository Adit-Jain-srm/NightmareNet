"""Unit tests for NightmareNet checkpoint saving, offset scheduling, and training resume."""

import os

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from nightmarenet.training.scheduler import CyclicScheduler
from nightmarenet.training.trainer import Trainer


@pytest.fixture(scope="module")
def shared_model_and_tokenizer():
    """Returns a tiny GPT-2 model and tokenizer to keep tests fast and memory-efficient."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Configuration for a tiny, zero-resource model
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=4,
        n_inner=4,
        vocab_size=len(tokenizer),
    )
    model = transformers.GPT2LMHeadModel(config)
    return model, tokenizer


def _make_tiny_dataset(n: int = 10) -> Dataset:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Paris is the capital of France and a major city.",
    ]
    return Dataset.from_dict({"text": [texts[i % len(texts)] for i in range(n)]})


def _tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 32):
    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = dataset.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


@pytest.fixture
def minimal_config(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    checkpoint_dir.mkdir()
    log_dir.mkdir()
    return {
        "model": {
            "name": "gpt2",
            "type": "causal_lm",
            "max_length": 32,
            "device": "cpu",
        },
        "dataset": {
            "text_column": "text",
            "max_samples": 10,
        },
        "training": {
            "wake_epochs": 1,
            "dream_epochs": 1,
            "nightmare_epochs": 1,
            "num_cycles": 2,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "save_every_phase": True,
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
        },
        "distortion": {
            "dream_strength": 0.25,
            "nightmare_strength": 0.8,
        },
        "compression": {
            "pruning_ratio": 0.1,
            "pruning_method": "magnitude",
        },
        "evaluation": {
            "metrics": ["recall"],
        },
        "tracking": {"backend": "none"},
        "seed": 42,
    }


def test_scheduler_resume_offset():
    """Test that CyclicScheduler correctly skips to the next phase when starting with offset."""
    scheduler = CyclicScheduler(
        num_cycles=3,
        wake_epochs=1,
        dream_epochs=1,
        nightmare_epochs=1,
        compression_rounds=1,
        start_cycle=0,
        start_phase="compress",
    )
    phases = list(scheduler)
    # Remaining: Cycle 1 wake, dream, nightmare, compress; Cycle 2 wake, dream, nightmare, compress
    assert len(phases) == 8
    assert phases[0] == (1, "wake", 1)
    assert phases[-1] == (2, "compress", 1)

    scheduler2 = CyclicScheduler(
        num_cycles=3,
        wake_epochs=1,
        dream_epochs=1,
        nightmare_epochs=1,
        compression_rounds=1,
        start_cycle=1,
        start_phase="dream",
    )
    phases2 = list(scheduler2)
    # Remaining: Cycle 1 nightmare, compress; Cycle 2 wake, dream, nightmare, compress
    assert len(phases2) == 6
    assert phases2[0] == (1, "nightmare", 1)


def test_trainer_save_and_load_state(minimal_config, shared_model_and_tokenizer):
    """Test that checkpoint saving preserves optimizer, scaler, scheduler state, and history."""
    model, tokenizer = shared_model_and_tokenizer
    trainer = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)

    # Checkpoint path
    path = os.path.join(trainer.checkpoint_dir, "default_run", "cycle-0-wake")

    # Let's populate history and save checkpoint
    trainer.history = [{"phase": "wake", "avg_loss": 1.23, "cycle": 0}]
    trainer._save_checkpoint(cycle=0, phase="wake")

    state_file = os.path.join(path, "training_state.pt")
    assert os.path.exists(state_file)
    assert os.path.exists(os.path.join(path, "config.json"))

    # Load state directly
    state = torch.load(state_file, map_location="cpu")
    assert "optimizer_state_dict" in state
    assert state["cycle"] == 0
    assert state["phase"] == "wake"
    assert state["history"] == [{"phase": "wake", "avg_loss": 1.23, "cycle": 0}]
    assert "metadata" in state
    assert state["metadata"]["trainer_class"] == "Trainer"


def test_trainer_resume_execution(minimal_config, shared_model_and_tokenizer):
    """Test training resume end-to-end with a dummy loop."""
    model, tokenizer = shared_model_and_tokenizer

    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    # Step 1: Initialize first trainer run
    trainer1 = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)
    trainer1.history = [{"phase": "wake", "avg_loss": 2.5, "cycle": 0}]

    # Save checkpoint manually at cycle 0 wake
    trainer1._save_checkpoint(cycle=0, phase="wake")
    checkpoint_path = os.path.join(trainer1.checkpoint_dir, "default_run", "cycle-0-wake")

    # Step 2: Create second trainer configured to resume
    resume_config = minimal_config.copy()
    resume_config["training"] = minimal_config["training"].copy()
    resume_config["training"]["resume_from"] = checkpoint_path

    trainer2 = Trainer(
        config=resume_config,
        model=model,
        tokenizer=tokenizer,
    )

    # Run train with short loaders
    history = trainer2.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
    )

    # Verify training continued from dream phase of cycle 0 (index 1 of Cycle 0)
    # Remaining phases of the run with 2 cycles total:
    # Cycle 0: dream, nightmare, compress
    # Cycle 1: wake, dream, nightmare, compress
    # Total new phases: 7
    # Original history from trainer1: 1 phase
    # Total history: 8 phases
    assert len(history) == 8
    assert history[0]["phase"] == "wake"
    assert history[0]["avg_loss"] == 2.5
    assert history[1]["phase"] == "dream"
    assert history[-1]["phase"] == "compress"


def test_trainer_resume_corrupted_phase(minimal_config, shared_model_and_tokenizer):
    """Test that a ValueError is raised if the checkpoint phase is corrupted or unknown."""
    model, tokenizer = shared_model_and_tokenizer
    trainer = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)
    path = os.path.join(trainer.checkpoint_dir, "default_run", "cycle-0-wake")

    # Save a checkpoint with a corrupted phase
    trainer.history = [{"phase": "wake", "avg_loss": 1.0, "cycle": 0}]
    trainer._save_checkpoint(cycle=0, phase="wake")
    state_file = os.path.join(path, "training_state.pt")

    state = torch.load(state_file, map_location="cpu")
    state["phase"] = "corrupted_phase_name"
    torch.save(state, state_file)

    # Recalculate checksum of training_state.pt and update metadata.json to pass check
    import json

    from nightmarenet.distributed.checkpoint import compute_file_sha256

    new_hash = compute_file_sha256(state_file)
    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["file_hashes"]["training_state.pt"] = new_hash
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Attempt to load state in a new trainer should raise ValueError
    resume_config = minimal_config.copy()
    resume_config["training"] = minimal_config["training"].copy()
    resume_config["training"]["resume_from"] = path

    # We mock loaders to trigger train load
    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    trainer2 = Trainer(
        config=resume_config,
        model=model,
        tokenizer=tokenizer,
    )
    with pytest.raises(ValueError, match="Corrupted start_phase 'corrupted_phase_name'"):
        trainer2.train(
            train_dataloader=loader,
            dream_dataloader=loader,
            nightmare_dataloader=loader,
            val_dataloader=loader,
        )


def test_amp_scaler_save_load(minimal_config, shared_model_and_tokenizer):
    """Test that AMP scaler state is correctly saved and loaded when use_amp is True."""
    model, tokenizer = shared_model_and_tokenizer
    config = minimal_config.copy()
    config["training"] = minimal_config["training"].copy()
    config["training"]["use_amp"] = True

    trainer = Trainer(config=config, model=model, tokenizer=tokenizer)
    trainer.use_amp = True
    trainer.scaler = torch.cuda.amp.GradScaler()

    # Mock state dict to simulate a scaler state
    trainer.scaler.state_dict = lambda: {"scale": 128.0, "growth_tracker": 1}

    path = os.path.join(trainer.checkpoint_dir, "default_run", "cycle-0-wake")
    trainer._save_checkpoint(cycle=0, phase="wake")
    state_file = os.path.join(path, "training_state.pt")
    assert os.path.exists(state_file)

    # Load and verify
    state = torch.load(state_file, map_location="cpu")
    assert "scaler_state_dict" in state
    assert state["scaler_state_dict"]["scale"] == 128.0

    # Test loading
    resume_config = config.copy()
    resume_config["training"]["resume_from"] = path
    trainer2 = Trainer(
        config=resume_config,
        model=model,
        tokenizer=tokenizer,
    )
    trainer2.use_amp = True
    trainer2.scaler = torch.cuda.amp.GradScaler()

    # Mock load_state_dict to capture what was loaded
    loaded_scale = None

    def mock_load_state_dict(sd):
        nonlocal loaded_scale
        loaded_scale = sd.get("scale")

    trainer2.scaler.load_state_dict = mock_load_state_dict

    # We mock loaders to trigger train load
    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    trainer2.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
    )

    # Scale should be loaded from checkpoint
    assert loaded_scale == 128.0


def test_chained_resume_execution(minimal_config, shared_model_and_tokenizer):
    """Test that multiple sequential resumes (chaining) work and accumulate history correctly."""
    model, tokenizer = shared_model_and_tokenizer

    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    # Start config with 3 cycles
    config = minimal_config.copy()
    config["training"] = minimal_config["training"].copy()
    config["training"]["num_cycles"] = 3

    # Run 1: Start from scratch, run first phase (wake) and stop (interrupt)
    trainer1 = Trainer(config=config, model=model, tokenizer=tokenizer)

    def on_progress1(event):
        if event["status"] == "phase_end" and event["phase"] == "wake" and event["cycle"] == 0:
            trainer1._interrupted = True

    trainer1.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
        on_progress=on_progress1,
    )
    assert len(trainer1.history) == 1
    assert trainer1.history[0]["phase"] == "wake"
    chk_a = os.path.join(trainer1.checkpoint_dir, "default_run", "cycle-0-wake")
    assert os.path.exists(chk_a)

    # Run 2: Resume from A, run second phase (dream) and stop (interrupt)
    config2 = config.copy()
    config2["training"] = config["training"].copy()
    config2["training"]["resume_from"] = chk_a
    trainer2 = Trainer(
        config=config2,
        model=model,
        tokenizer=tokenizer,
    )

    def on_progress2(event):
        if event["status"] == "phase_end" and event["phase"] == "dream" and event["cycle"] == 0:
            trainer2._interrupted = True

    trainer2.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
        on_progress=on_progress2,
    )
    assert len(trainer2.history) == 2
    assert trainer2.history[0]["phase"] == "wake"
    assert trainer2.history[1]["phase"] == "dream"
    chk_b = os.path.join(trainer2.checkpoint_dir, "default_run", "cycle-0-dream")
    assert os.path.exists(chk_b)

    # Run 3: Resume from B, run to completion (cycles = 3, so remaining 10 phases)
    config3 = config.copy()
    config3["training"] = config["training"].copy()
    config3["training"]["resume_from"] = chk_b
    trainer3 = Trainer(
        config=config3,
        model=model,
        tokenizer=tokenizer,
    )

    history3 = trainer3.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
    )

    # Total phases in 3 cycles is 12 (3 * 4)
    assert len(history3) == 12
    assert history3[0]["phase"] == "wake"
    assert history3[1]["phase"] == "dream"
    assert history3[2]["phase"] == "nightmare"
    assert history3[-1]["phase"] == "compress"


def test_adaptive_scheduler_resume(minimal_config, shared_model_and_tokenizer):
    """Test that AdaptiveScheduler is correctly offset during resume."""
    model, tokenizer = shared_model_and_tokenizer
    config = minimal_config.copy()
    config["training"] = minimal_config["training"].copy()
    config["training"]["early_stopping"] = True
    config["training"]["num_cycles"] = 2

    # Check scheduler type is AdaptiveScheduler
    trainer1 = Trainer(config=config, model=model, tokenizer=tokenizer)
    assert trainer1.scheduler.__class__.__name__ == "AdaptiveScheduler"

    # Save checkpoint manually at cycle 0 dream
    trainer1.history = [
        {"phase": "wake", "avg_loss": 2.0, "cycle": 0},
        {"phase": "dream", "avg_loss": 1.8, "cycle": 0},
    ]
    trainer1._save_checkpoint(cycle=0, phase="dream")
    checkpoint_path = os.path.join(trainer1.checkpoint_dir, "default_run", "cycle-0-dream")

    # Resume with adaptive scheduler
    resume_config = config.copy()
    resume_config["training"] = config["training"].copy()
    resume_config["training"]["resume_from"] = checkpoint_path
    trainer2 = Trainer(
        config=resume_config,
        model=model,
        tokenizer=tokenizer,
    )

    # We mock loaders to trigger train load
    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    # Let's interrupt immediately so it doesn't run the epochs
    def on_progress2(event):
        trainer2._interrupted = True

    trainer2.train(
        train_dataloader=loader,
        dream_dataloader=loader,
        nightmare_dataloader=loader,
        val_dataloader=loader,
        on_progress=on_progress2,
    )

    # Check that offsets propagated to base_scheduler
    assert trainer2.scheduler.base_scheduler.start_cycle == 0
    assert trainer2.scheduler.base_scheduler.start_phase == "dream"

    # Verify the remaining schedule (2 cycles * 4 phases = 8 total. Skipping first 2)
    # Remaining: Cycle 0 nightmare, compress; Cycle 1 wake, dream, nightmare, compress
    remaining_phases = list(trainer2.scheduler.base_scheduler)
    assert len(remaining_phases) == 6
    assert remaining_phases[0] == (0, "nightmare", 1)


def test_optimizer_param_group_mismatch(minimal_config, shared_model_and_tokenizer):
    """Test warning logging and skipping optimizer load if param groups mismatch."""
    model, tokenizer = shared_model_and_tokenizer
    trainer1 = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)

    # Save checkpoint manually at cycle 0 wake
    trainer1.history = [{"phase": "wake", "avg_loss": 2.5, "cycle": 0}]
    trainer1._save_checkpoint(cycle=0, phase="wake")
    checkpoint_path = os.path.join(trainer1.checkpoint_dir, "default_run", "cycle-0-wake")

    # Now create a new trainer and mock its optimizer param groups to have a different count
    resume_config = minimal_config.copy()
    resume_config["training"] = minimal_config["training"].copy()
    resume_config["training"]["resume_from"] = checkpoint_path

    trainer2 = Trainer(
        config=resume_config,
        model=model,
        tokenizer=tokenizer,
    )
    # Manually append a dummy param group to change group count
    trainer2.optimizer.add_param_group({"params": []})

    base_ds = _make_tiny_dataset(4)
    train_ds = _tokenize_dataset(base_ds, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)

    # Interrupt immediately
    def on_progress(event):
        trainer2._interrupted = True

    import unittest.mock as mock

    from nightmarenet.training.trainer import logger as trainer_logger

    with mock.patch.object(trainer_logger, "warning") as mock_warning:
        trainer2.train(
            train_dataloader=loader,
            dream_dataloader=loader,
            nightmare_dataloader=loader,
            val_dataloader=loader,
            on_progress=on_progress,
        )

    # Verify warning was logged
    mock_warning.assert_called_once()
    call_args = mock_warning.call_args[0]
    assert "Optimizer param group count mismatch" in call_args[0]


def test_checkpoint_version_compatibility(minimal_config, shared_model_and_tokenizer):
    """Test that validating checkpoint integrity raises ValueError on incompatible version."""
    import json

    from nightmarenet.distributed.checkpoint import validate_checkpoint_integrity

    model, tokenizer = shared_model_and_tokenizer
    trainer = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)
    trainer.history = [{"phase": "wake", "avg_loss": 2.5, "cycle": 0}]
    trainer._save_checkpoint(cycle=0, phase="wake")
    checkpoint_path = os.path.join(trainer.checkpoint_dir, "default_run", "cycle-0-wake")

    # Modify version in metadata.json to be incompatible
    meta_path = os.path.join(checkpoint_path, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["version"] = "999.0.0"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Attempting to validate should raise ValueError
    with pytest.raises(ValueError, match="Incompatible checkpoint version 999.0.0"):
        validate_checkpoint_integrity(checkpoint_path, minimal_config)


def test_checkpoint_checksum_integrity_validation(minimal_config, shared_model_and_tokenizer):
    """Test that modifying saved file contents triggers integrity validation failure."""
    from nightmarenet.distributed.checkpoint import validate_checkpoint_integrity

    model, tokenizer = shared_model_and_tokenizer
    trainer = Trainer(config=minimal_config, model=model, tokenizer=tokenizer)
    trainer.history = [{"phase": "wake", "avg_loss": 2.5, "cycle": 0}]
    trainer._save_checkpoint(cycle=0, phase="wake")
    checkpoint_path = os.path.join(trainer.checkpoint_dir, "default_run", "cycle-0-wake")

    # Corrupt optimizer.pt file
    optimizer_pt = os.path.join(checkpoint_path, "optimizer.pt")
    with open(optimizer_pt, "ab") as f:
        f.write(b"corrupted_tail_bytes")

    # Validation should raise ValueError due to checksum mismatch
    with pytest.raises(ValueError, match="Integrity check failed: Checksum mismatch"):
        validate_checkpoint_integrity(checkpoint_path, minimal_config)
