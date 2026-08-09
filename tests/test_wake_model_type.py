"""Regression tests: model_type wiring in WakePhase.

Verifies that both causal_lm (default) and seq_classification paths
work without crashing, and that labels are set correctly.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nightmarenet.training.phases import WakePhase


def _make_dummy_dataloader(include_labels: bool = False, batch_size: int = 2):
    """Create a tiny dataloader with fake tokenized data."""
    input_ids = torch.randint(0, 100, (4, 8))
    attention_mask = torch.ones(4, 8, dtype=torch.long)
    if include_labels:
        labels = torch.randint(0, 2, (4,))
        ds = TensorDataset(input_ids, attention_mask, labels)
    else:
        ds = TensorDataset(input_ids, attention_mask)

    def collate(batch):
        if include_labels:
            ids, mask, labs = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(labs),
            }
        ids, mask = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(mask),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


class _TinyCausalModel(torch.nn.Module):
    """Minimal model that returns a loss shaped like a causal LM."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.linear(input_ids.float())
        loss = logits.sum() * 0.001
        return type("Output", (), {"loss": loss, "logits": logits})()


class _TinySeqClassModel(torch.nn.Module):
    """Minimal model that returns a loss shaped like a seq classifier."""

    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = input_ids.float()  # (batch, seq_len=8)
        logits = self.classifier(hidden)  # (batch, 2)
        loss = torch.nn.functional.cross_entropy(
            logits, labels if labels is not None else torch.zeros(logits.size(0), dtype=torch.long)
        )
        return type("Output", (), {"loss": loss, "logits": logits})()


def test_wake_causal_lm_assigns_labels_from_input_ids():
    """Default causal_lm mode should set labels = input_ids."""
    model = _TinyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dl = _make_dummy_dataloader(include_labels=False)

    phase = WakePhase(
        model=model,
        optimizer=optimizer,
        config={},
        device="cpu",
        model_type="causal_lm",
    )
    result = phase.run(dl, num_epochs=1)
    assert result["phase"] == "wake"
    assert result["total_steps"] > 0
    assert result["avg_loss"] != 0.0


def test_wake_seq_classification_preserves_existing_labels():
    """seq_classification mode should NOT overwrite batch['labels']."""
    model = _TinySeqClassModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dl = _make_dummy_dataloader(include_labels=True)

    phase = WakePhase(
        model=model,
        optimizer=optimizer,
        config={},
        device="cpu",
        model_type="seq_classification",
    )
    result = phase.run(dl, num_epochs=1)
    assert result["phase"] == "wake"
    assert result["total_steps"] > 0


def test_wake_zero_epochs_skips_gracefully():
    """num_epochs=0 should return immediately without training."""
    model = _TinyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dl = _make_dummy_dataloader()

    phase = WakePhase(
        model=model,
        optimizer=optimizer,
        config={},
        device="cpu",
    )
    result = phase.run(dl, num_epochs=0)
    assert result["total_steps"] == 0
    assert result["avg_loss"] == 0.0
