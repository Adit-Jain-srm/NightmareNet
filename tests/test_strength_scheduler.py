import os

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nightmarenet.data.generator import DreamDatasetGenerator, NightmareDatasetGenerator
from nightmarenet.training.trainer import Trainer


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch):
    """Prevent HF hub connections during tests."""
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _make_tiny_dataset(n: int = 10) -> Dataset:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Paris is the capital of France and a major city.",
    ]
    return Dataset.from_dict({"text": [texts[i % len(texts)] for i in range(n)]})

def _tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 16):
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

def _make_dataloader(dataset: Dataset, tokenizer, batch_size: int = 2):
    tokenized = _tokenize_dataset(dataset, tokenizer)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)

def _get_mock_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True)

    # Mock save_pretrained to write a tiny dummy file to prevent disk space issues
    def mock_save_pretrained(save_directory, *args, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write("{}")
        torch.save({}, os.path.join(save_directory, "model.pt"))

    model.save_pretrained = mock_save_pretrained
    if hasattr(model, "config") and model.config is not None:
        model.config.save_pretrained = lambda *args, **kwargs: None

    return model, tokenizer

class TestStrengthScheduler:
    """Test suite for per-cycle distortion strength scheduling."""

    def test_strength_scheduling_enabled(self, tmp_path):
        """Test that distortion strengths follow the schedule when enabled."""
        model, tokenizer = _get_mock_model_and_tokenizer()
        base_ds = _make_tiny_dataset(4)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 3,
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 0,
                "lr_schedule": "none",
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "log_dir": str(tmp_path / "logs"),
            },
            "distortion": {
                "schedule_across_cycles": True,
                "strength_min": 0.2,
                "strength_max": 0.9,
                "dream_strength": 0.25,
                "nightmare_strength": 0.8,
            },
            "compression": {
                "finetune_after_prune": False,
            }
        }

        # Track the strength values set on generators across cycles
        dream_strengths = []
        nightmare_strengths = []

        # Create generators and hook into their strength property setters
        dream_gen = DreamDatasetGenerator(strength=0.25)
        nightmare_gen = NightmareDatasetGenerator(strength=0.8)

        # Mock the generate methods to record strength and return a dummy dataset
        original_dream_generate = dream_gen.generate
        original_nightmare_generate = nightmare_gen.generate

        def mock_dream_generate(dataset):
            dream_strengths.append(dream_gen.strength)
            return original_dream_generate(dataset)

        def mock_nightmare_generate(dataset):
            nightmare_strengths.append(nightmare_gen.strength)
            return original_nightmare_generate(dataset)

        dream_gen.generate = mock_dream_generate
        nightmare_gen.generate = mock_nightmare_generate

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
            dream_generator=dream_gen,
            nightmare_generator=nightmare_gen,
            dream_base_dataset=base_ds,
            nightmare_base_dataset=base_ds,
        )

        # Formula: 0.2 + (0.9 - 0.2) * cycle / (3 - 1)
        # Cycle 0: 0.2
        # Cycle 1: 0.55
        # Cycle 2: 0.9
        assert len(dream_strengths) == 3
        assert len(nightmare_strengths) == 3

        # Test values are close to formula computed values
        assert abs(dream_strengths[0] - 0.2) < 1e-5
        assert abs(dream_strengths[1] - 0.55) < 1e-5
        assert abs(dream_strengths[2] - 0.9) < 1e-5

        assert abs(nightmare_strengths[0] - 0.2) < 1e-5
        assert abs(nightmare_strengths[1] - 0.55) < 1e-5
        assert abs(nightmare_strengths[2] - 0.9) < 1e-5

    def test_strength_scheduling_disabled(self, tmp_path):
        """Test that distortion strengths remain static when disabled."""
        model, tokenizer = _get_mock_model_and_tokenizer()
        base_ds = _make_tiny_dataset(4)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 2,
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 0,
                "lr_schedule": "none",
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "log_dir": str(tmp_path / "logs"),
            },
            "distortion": {
                "schedule_across_cycles": False,
                "strength_min": 0.2,
                "strength_max": 0.9,
                "dream_strength": 0.25,
                "nightmare_strength": 0.8,
            },
            "compression": {
                "finetune_after_prune": False,
            }
        }

        dream_gen = DreamDatasetGenerator(strength=0.25)
        nightmare_gen = NightmareDatasetGenerator(strength=0.8)

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
            dream_generator=dream_gen,
            nightmare_generator=nightmare_gen,
            dream_base_dataset=base_ds,
            nightmare_base_dataset=base_ds,
        )

        # Stays at configured default strengths
        assert dream_gen.strength == 0.25
        assert nightmare_gen.strength == 0.8
