"""Focused tests for model-aware learned adversarial distortions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.learned import LearnedAdversarialGenerator


class TinyEncoding(dict):
    """Minimal BatchEncoding-like object with word alignment support."""

    def __init__(self, input_ids: torch.Tensor, word_ids: list[Optional[int]]):
        super().__init__(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[Optional[int]]:
        del batch_index
        return self._word_ids


class TinyTokenizer:
    """Offline tokenizer used to test gradient and attention strategies."""

    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[MASK]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "epsilon": 8,
        "zeta": 9,
    }
    inverse_vocab = {value: key for key, value in vocab.items()}
    mask_token = "[MASK]"
    mask_token_id = vocab[mask_token]
    all_special_ids = [0, 1, 2, 3]

    def __call__(
        self,
        text_or_words,
        *,
        is_split_into_words: bool = False,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
    ) -> TinyEncoding:
        del return_tensors, truncation, max_length
        if is_split_into_words:
            words = list(text_or_words)
        else:
            words = str(text_or_words).split()

        ids = [self.vocab["[CLS]"]]
        word_ids: list[Optional[int]] = [None]
        for index, word in enumerate(words):
            token_key = word if word in self.vocab else word.casefold()
            ids.append(self.vocab.get(token_key, self.vocab["delta"]))
            word_ids.append(index)
        ids.append(self.vocab["[SEP]"])
        word_ids.append(None)
        return TinyEncoding(torch.tensor([ids], dtype=torch.long), word_ids)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        words = []
        for token_id in token_ids:
            word = self.inverse_vocab[int(token_id)]
            if skip_special_tokens and int(token_id) in self.all_special_ids:
                continue
            words.append(word)
        return " ".join(words)


class TinyTargetModel(nn.Module):
    """Toy target whose last input word has the largest embedding gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(len(TinyTokenizer.vocab), 3)
        with torch.no_grad():
            weights = torch.zeros(len(TinyTokenizer.vocab), 3)
            weights[:, 0] = torch.arange(len(TinyTokenizer.vocab), dtype=torch.float32)
            weights[:, 1] = torch.arange(len(TinyTokenizer.vocab), dtype=torch.float32) / 10
            self.embeddings.weight.copy_(weights)
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, inputs_embeds, attention_mask=None):
        del attention_mask
        self.forward_calls += 1
        sequence_length = inputs_embeds.shape[1]
        positional_weight = torch.arange(
            1,
            sequence_length + 1,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).view(1, sequence_length, 1)
        pooled = (inputs_embeds * positional_weight).sum(dim=1)
        logits = torch.stack(
            (pooled[:, 0], pooled[:, 0] + 0.01 * pooled[:, 1]),
            dim=-1,
        )
        return SimpleNamespace(logits=logits)


class TinyAttentionModel(nn.Module):
    """Toy masked LM that prioritises the first word and predicts epsilon."""

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            len(TinyTokenizer.vocab),
            dtype=torch.float32,
        )
        logits[..., TinyTokenizer.vocab["epsilon"]] = 10.0
        logits[..., TinyTokenizer.vocab["zeta"]] = 9.0

        attentions = None
        if output_attentions:
            attention = torch.zeros(batch_size, 1, sequence_length, sequence_length)
            attention[..., 1] = 10.0
            attentions = (attention,)
        return SimpleNamespace(logits=logits, attentions=attentions)


def make_attention_generator(seed: int = 42) -> LearnedAdversarialGenerator:
    """Create an attention generator without downloading a Hugging Face model."""
    generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
        seed=seed,
    )
    generator.strategy = "attention"
    generator._tokenizer = TinyTokenizer()
    generator._model = TinyAttentionModel()
    generator._available = True
    return generator


def test_gradient_importance_ranks_target_sensitive_word_highest() -> None:
    generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
    )

    analysis = generator._gradient_analysis("alpha beta gamma")

    assert analysis.scores[2] > analysis.scores[1] > analysis.scores[0]


def test_gradient_projection_changes_most_sensitive_word() -> None:
    generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
        seed=7,
    )

    result = generator.generate("alpha beta gamma", strength=1.0)

    assert result == "alpha beta alpha"


def test_gradient_and_attention_strategies_produce_different_outputs() -> None:
    gradient_generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
        seed=11,
    )
    attention_generator = make_attention_generator(seed=11)

    text = "alpha beta gamma"
    gradient_result = gradient_generator.generate(text, strength=1.0)
    attention_result = attention_generator.generate(text, strength=1.0)

    assert gradient_result != attention_result
    assert gradient_result == "alpha beta alpha"
    assert attention_result == "epsilon beta gamma"


def test_gradient_generation_is_deterministic_with_fixed_seed() -> None:
    first = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
        seed=99,
    )
    second = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=TinyTargetModel(),
        target_tokenizer=TinyTokenizer(),
        seed=99,
    )

    assert first.generate("alpha beta gamma", 1.0) == second.generate(
        "alpha beta gamma",
        1.0,
    )


def test_cache_is_partitioned_by_training_cycle() -> None:
    model = TinyTargetModel()
    generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=model,
        target_tokenizer=TinyTokenizer(),
        cache_enabled=True,
    )

    first = generator.generate("alpha beta gamma", 1.0, cycle_id=0)
    calls_after_first = model.forward_calls
    cached = generator.generate("alpha beta gamma", 1.0, cycle_id=0)
    calls_after_cached = model.forward_calls
    next_cycle = generator.generate("alpha beta gamma", 1.0, cycle_id=1)

    assert first == cached == next_cycle
    assert calls_after_first == calls_after_cached
    assert model.forward_calls > calls_after_cached
    assert len(generator.get_cached_examples(0)) == 1
    assert len(generator.get_cached_examples(1)) == 1


def test_gradient_strategy_falls_back_to_attention_without_target_model() -> None:
    generator = make_attention_generator()
    generator.strategy = "gradient"
    generator.target_model = None
    generator.target_tokenizer = None

    result = generator.generate("alpha beta gamma", strength=1.0)

    assert result == "epsilon beta gamma"


def test_learned_zero_keeps_disabled_path_unchanged(monkeypatch) -> None:
    def fail_if_created(*args, **kwargs):
        del args, kwargs
        raise AssertionError("learned generator should not be created")

    monkeypatch.setattr(
        "nightmarenet.distortions.adversarial._get_learned_generator",
        fail_if_created,
    )
    config = {
        "contradiction": 0.0,
        "ambiguity": 0.0,
        "cross_domain": 0.0,
        "misleading_context": 0.0,
        "learned": 0.0,
        "learned_strategy": "gradient",
    }
    text = "alpha beta gamma"

    assert apply_adversarial_distortions(text, 0.5, config) == text


def test_gradient_attack_does_not_accumulate_parameter_gradients() -> None:
    model = TinyTargetModel()
    model.train()
    generator = LearnedAdversarialGenerator(
        strategy="gradient",
        target_model=model,
        target_tokenizer=TinyTokenizer(),
        cache_enabled=False,
    )

    generator.generate("alpha beta gamma", strength=1.0)

    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())
