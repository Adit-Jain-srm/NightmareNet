"""Tests for RSLAD-style distillation in the compression phase."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM


@pytest.fixture
def tiny_model():
    """Create a small GPT2 model from config (no download)."""
    from transformers import GPT2Config

    config = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
    model = AutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def tiny_dataloader():
    """Create a minimal dataloader with fake token IDs."""
    input_ids = torch.randint(0, 100, (8, 16))
    attention_mask = torch.ones_like(input_ids)
    dataset = TensorDataset(input_ids, attention_mask)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": mask}

    return DataLoader(dataset, batch_size=4, collate_fn=collate)


class TestDistillationLoss:
    def test_distillation_loss_decreases(self, tiny_model, tiny_dataloader):
        """Run 2 epochs of distillation and verify loss decreases."""
        import copy

        from nightmarenet.compression.distillation import run_distillation

        teacher = copy.deepcopy(tiny_model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = tiny_model
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        # Run 1 epoch
        result_1 = run_distillation(
            teacher=teacher,
            student=student,
            dataloader=tiny_dataloader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=1,
            temperature=4.0,
            alpha=0.7,
        )

        # Run another epoch
        result_2 = run_distillation(
            teacher=teacher,
            student=student,
            dataloader=tiny_dataloader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=1,
            temperature=4.0,
            alpha=0.7,
        )

        assert result_1["distillation_loss"] > 0
        assert result_2["distillation_loss"] > 0
        # Loss should decrease (or at least not increase significantly)
        assert result_2["distillation_loss"] <= result_1["distillation_loss"] * 1.1


class TestDistillationDisabled:
    def test_distillation_disabled_noop(self, tiny_model, tiny_dataloader):
        """When distillation=false, no teacher should be created."""
        from nightmarenet.training.phases import CompressionPhase

        config = {
            "pruning_ratio": 0.1,
            "pruning_method": "magnitude",
            "distillation": False,
            "finetune_after_prune": False,
        }

        phase = CompressionPhase(
            model=tiny_model,
            config=config,
            device="cpu",
        )

        with patch("nightmarenet.compression.distillation.run_distillation") as mock_distill:
            phase.run(
                dataloader=tiny_dataloader,
                optimizer=torch.optim.SGD(tiny_model.parameters(), lr=0.01),
            )

        mock_distill.assert_not_called()


class TestDistillationGatedOnPruning:
    def test_distillation_gated_on_pruning_success(self, tiny_model, tiny_dataloader):
        """If pruning fails, distillation should be skipped."""
        from nightmarenet.training.phases import CompressionPhase

        config = {
            "pruning_ratio": 0.1,
            "pruning_method": "magnitude",
            "distillation": True,
            "finetune_after_prune": False,
        }

        phase = CompressionPhase(
            model=tiny_model,
            config=config,
            device="cpu",
        )

        # Mock pruner to raise an exception
        with patch(
            "nightmarenet.compression.pruning.MagnitudePruner.apply",
            side_effect=RuntimeError("Pruning failed"),
        ):
            with patch("nightmarenet.compression.distillation.run_distillation") as mock_distill:
                result = phase.run(
                    dataloader=tiny_dataloader,
                    optimizer=torch.optim.SGD(tiny_model.parameters(), lr=0.01),
                )

        mock_distill.assert_not_called()
        assert result["pruned_params"] == 0
