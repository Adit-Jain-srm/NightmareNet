import os

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nightmarenet.training.phases import NightmarePhase
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

class TestLRScheduler:
    """Test suite for learning rate warmup + cosine decay scheduler."""

    def test_lr_schedule_instantiation(self, tmp_path):
        """Test that get_cosine_schedule_with_warmup is instantiated correctly."""
        model, tokenizer = _get_mock_model_and_tokenizer()

        base_ds = _make_tiny_dataset(4)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 1,
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 2,
                "lr_schedule": "linear_warmup_cosine",
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "log_dir": str(tmp_path / "logs"),
            },
            "compression": {
                "finetune_after_prune": True,
                "finetune_epochs": 1,
            }
        }

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
        )

        assert trainer.lr_scheduler is not None
        from torch.optim.lr_scheduler import LambdaLR
        assert isinstance(trainer.lr_scheduler, LambdaLR)

    def test_lr_warmup_and_decay_shape(self, tmp_path):
        """Test the learning rate starts near 0, peaks at warmup, and decays."""
        model, tokenizer = _get_mock_model_and_tokenizer()

        base_ds = _make_tiny_dataset(8)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 1,
                "wake_epochs": 2,
                "dream_epochs": 0,
                "nightmare_epochs": 0,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "warmup_steps": 4,
                "lr_schedule": "linear_warmup_cosine",
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "log_dir": str(tmp_path / "logs"),
            },
            "compression": {
                "finetune_after_prune": False,
            }
        }

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        # Track learning rate values
        lrs = []
        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
            on_progress=lambda progress: lrs.append(trainer.optimizer.param_groups[0]["lr"])
        )

        assert trainer.lr_scheduler is not None

        base_lr = 1e-4
        lrs_sampled = []

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=8
        )

        lrs_sampled.append(optimizer.param_groups[0]["lr"])
        for _ in range(8):
            optimizer.step()
            sched.step()
            lrs_sampled.append(optimizer.param_groups[0]["lr"])

        assert lrs_sampled[0] == 0.0
        assert lrs_sampled[1] < base_lr
        assert abs(lrs_sampled[4] - base_lr) < 1e-7
        assert lrs_sampled[5] < lrs_sampled[4]
        assert lrs_sampled[8] < lrs_sampled[5]

    def test_lr_scheduler_backward_compatibility(self, tmp_path):
        """Test that lr_schedule = 'none' disables the scheduler even with warmup_steps > 0."""
        model, tokenizer = _get_mock_model_and_tokenizer()

        base_ds = _make_tiny_dataset(4)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 1,
                "wake_epochs": 1,
                "dream_epochs": 0,
                "nightmare_epochs": 0,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 100,
                "lr_schedule": "none",
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "log_dir": str(tmp_path / "logs"),
            },
            "compression": {
                "finetune_after_prune": False,
            }
        }

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
        )

        assert trainer.lr_scheduler is None

    def test_nightmare_lr_composure(self):
        """Test Nightmare phase LR multiplier composition and recovery."""
        model, tokenizer = _get_mock_model_and_tokenizer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=10
        )

        base_ds = _make_tiny_dataset(4)
        loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        initial_lr = optimizer.param_groups[0]["lr"]

        phase = NightmarePhase(
            model=model,
            optimizer=optimizer,
            config={"max_grad_norm": 1.0, "gradient_accumulation_steps": 1},
            device="cpu",
            lr_multiplier=3.0,
            lr_scheduler=sched,
        )

        phase.run(loader, num_epochs=1)

        post_lr = optimizer.param_groups[0]["lr"]
        assert abs(post_lr - initial_lr) < 1e-7

    def test_checkpoint_save_and_restore(self, tmp_path):
        """Test scheduler state is successfully saved and loaded."""
        model, tokenizer = _get_mock_model_and_tokenizer()

        base_ds = _make_tiny_dataset(4)
        train_loader = _make_dataloader(base_ds, tokenizer, batch_size=2)

        checkpoint_dir = str(tmp_path / "ckpt")
        log_dir = str(tmp_path / "logs")

        config = {
            "seed": 42,
            "training": {
                "num_cycles": 1,
                "wake_epochs": 1,
                "dream_epochs": 0,
                "nightmare_epochs": 0,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "warmup_steps": 2,
                "lr_schedule": "linear_warmup_cosine",
                "gradient_accumulation_steps": 1,
                "save_every_phase": True,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
            },
            "compression": {
                "finetune_after_prune": False,
            }
        }

        model.state_dict = lambda *args, **kwargs: {}

        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )

        trainer.optimizer.state_dict = lambda: {}

        trainer.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
        )

        assert trainer.lr_scheduler is not None
        first_run_state = trainer.lr_scheduler.state_dict()

        # Let's create a new Trainer and resume from the saved checkpoint
        config_resume = dict(config)
        resume_path = os.path.join(checkpoint_dir, "default_run", "cycle-0-wake")
        config_resume["training"]["resume_from"] = resume_path

        model_resume, _ = _get_mock_model_and_tokenizer()
        model_resume.state_dict = lambda *args, **kwargs: {}

        trainer_resume = Trainer(
            model=model_resume,
            config=config_resume,
            tokenizer=tokenizer,
        )
        trainer_resume.optimizer.state_dict = lambda: {}

        trainer_resume.train(
            train_dataloader=train_loader,
            dream_dataloader=train_loader,
            nightmare_dataloader=train_loader,
        )

        assert trainer_resume.lr_scheduler is not None
        resumed_state = trainer_resume.lr_scheduler.state_dict()

        assert resumed_state["last_epoch"] == first_run_state["last_epoch"]
        assert resumed_state["base_lrs"] == first_run_state["base_lrs"]
