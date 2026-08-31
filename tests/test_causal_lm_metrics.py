"""Tests for causal LM perplexity-under-distortion metrics (issue #322)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from nightmarenet.evaluation.causal_lm import (
    evaluate_causal_lm_robustness,
    relative_ppl_degradation,
    texts_to_dataloader,
)


class TinyCausalLM(nn.Module):
    """Minimal causal LM stub with vocab logits for perplexity tests."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


class FakeTokenizer:
    """Deterministic char-ish tokenizer for unit tests (no HF download)."""

    def __call__(
        self,
        texts: List[str],
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 16,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        rows = []
        masks = []
        for text in texts:
            ids = [(ord(c) % 30) + 1 for c in text[: max_length - 1]] or [1]
            ids = ids[:max_length]
            pad = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad
            ids = ids + [0] * pad
            rows.append(ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_relative_ppl_degradation():
    assert relative_ppl_degradation(10.0, 15.0) == pytest.approx(0.5)
    assert relative_ppl_degradation(10.0, 8.0) == 0.0


def test_texts_to_dataloader_shapes():
    tok = FakeTokenizer()
    loader = texts_to_dataloader(tok, ["hello world", "foo bar"], max_length=16, batch_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_evaluate_causal_lm_robustness_three_types(monkeypatch):
    model = TinyCausalLM()
    tok = FakeTokenizer()
    texts = ["hello world this is a test", "another sample sentence here"]

    # Avoid importing heavy distortion stacks; identity maps keep PPL finite.
    monkeypatch.setattr(
        "nightmarenet.evaluation.causal_lm._build_distorter",
        lambda name, strength, seed=42: lambda text: text + f" {name}",
    )

    result = evaluate_causal_lm_robustness(
        model,
        tok,
        texts,
        distortion_types=("dream", "nightmare", "text"),
        strength=0.5,
        device="cpu",
        max_length=16,
        batch_size=2,
    )
    assert result["metric"] == "causal_lm_robustness"
    assert set(result["distortion_types"]) == {"dream", "nightmare", "text"}
    assert "clean_perplexity" in result
    assert len(result["per_distortion"]) == 3
    assert result["robustness_score"] > 0


def test_evaluate_requires_three_distortion_types():
    model = TinyCausalLM()
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="at least 3"):
        evaluate_causal_lm_robustness(
            model,
            tok,
            ["hello"],
            distortion_types=("dream", "text"),
            device="cpu",
        )


def test_build_distorter_seed_propagation():
    from nightmarenet.evaluation.causal_lm import _build_distorter

    text = "The quick brown fox jumps over the lazy dog and runs across the wide open field."
    fn1 = _build_distorter("dream", strength=0.5, seed=42)
    fn2 = _build_distorter("dream", strength=0.5, seed=42)
    fn3 = _build_distorter("dream", strength=0.5, seed=999)

    out1 = fn1(text)
    out2 = fn2(text)
    out3 = fn3(text)

    # Same seed produces identical distortion
    assert out1 == out2
    # Different seed produces different distortion output
    assert out1 != out3
