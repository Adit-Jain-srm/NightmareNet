"""Model-aware learned adversarial text distortions.

The gradient strategy scores tokens with the target model's embedding gradients and
selects replacement tokens by projecting those gradients onto the target embedding
matrix. When a target model is unavailable, the generator falls back to the original
attention-guided masked-language-model strategy.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"attention", "gradient"}


@dataclass
class _GradientAnalysis:
    """Per-word gradient information for a single text sample."""

    scores: list[float]
    gradients: list[Any]
    token_ids: list[list[int]]


class LearnedAdversarialGenerator:
    """Generate learned adversarial text substitutions.

    Args:
        model_name: Masked-LM name used by the attention strategy and as fallback.
        device: Device for the fallback masked LM.
        strength: Default distortion strength in ``[0, 1]``.
        target_model: Optional model being trained. Required for model-aware gradients.
        target_tokenizer: Optional tokenizer paired with ``target_model``.
        strategy: ``"gradient"`` for target-model-aware attacks or ``"attention"``
            for the legacy masked-LM method.
        cache_enabled: Cache generated examples per cycle.
        seed: Seed used by deterministic fallback operations.
        cache_max_entries: Maximum number of generated examples retained in memory.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cpu",
        strength: float = 0.5,
        target_model: Optional[Any] = None,
        target_tokenizer: Optional[Any] = None,
        strategy: str = "attention",
        cache_enabled: bool = True,
        seed: int = 42,
        cache_max_entries: int = 4096,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown learned adversarial strategy '{strategy}'. "
                f"Expected one of {sorted(_VALID_STRATEGIES)}."
            )
        if cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be greater than zero")

        self.model_name = model_name
        self.device = device
        self.strength = strength
        self.strategy = strategy
        self.cache_enabled = cache_enabled
        self.seed = seed
        self.cache_max_entries = cache_max_entries
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.cycle_id = 0

        self._model: Any = None
        self._tokenizer: Any = None
        self._available: Optional[bool] = None
        self._cache: OrderedDict[tuple[Any, ...], str] = OrderedDict()

        if self.strategy == "attention" or self.target_model is None:
            self._ensure_fallback_model()

    @property
    def gradient_available(self) -> bool:
        """Return whether the model-aware gradient strategy can be used."""
        return self.target_model is not None and self._gradient_tokenizer is not None

    @property
    def _gradient_tokenizer(self) -> Optional[Any]:
        return self.target_tokenizer or self._tokenizer

    def set_target_model(
        self,
        target_model: Optional[Any],
        target_tokenizer: Optional[Any] = None,
    ) -> None:
        """Update the model and tokenizer used by gradient attacks."""
        self.target_model = target_model
        if target_tokenizer is not None:
            self.target_tokenizer = target_tokenizer

    def set_cycle(self, cycle_id: int) -> None:
        """Set the cycle identifier used to partition generated-example cache entries."""
        self.cycle_id = int(cycle_id)

    def get_cached_examples(self, cycle_id: Optional[int] = None) -> list[str]:
        """Return cached generated examples, optionally restricted to one cycle."""
        # Cache keys are (cycle_id, model_id, text, strength, strategy).
        return [
            value for key, value in self._cache.items() if cycle_id is None or key[0] == cycle_id
        ]

    def clear_cache(self) -> None:
        """Remove all cached adversarial examples."""
        self._cache.clear()

    def _ensure_fallback_model(self) -> bool:
        """Lazily load the masked LM used by the legacy attention strategy."""
        if self._available is not None:
            return self._available

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._available = True
            logger.info("Loaded learned adversarial fallback model: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load adversarial fallback model '%s': %s. "
                "Random deterministic replacement will be used.",
                self.model_name,
                exc,
            )
            self._available = False
        return self._available

    def _rng_for(self, text: str, strength: float) -> random.Random:
        payload = f"{self.seed}\0{self.cycle_id}\0{strength:.8f}\0{text}".encode()
        deterministic_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(deterministic_seed)

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        return getattr(model, "module", model)

    @staticmethod
    def _word_ids(encoding: Any, word_count: int) -> list[Optional[int]]:
        try:
            word_ids = encoding.word_ids(batch_index=0)
        except (AttributeError, TypeError):
            try:
                word_ids = encoding.word_ids()
            except (AttributeError, TypeError):
                word_ids = None

        if word_ids is not None:
            return list(word_ids)

        input_ids = encoding["input_ids"][0]
        if len(input_ids) <= 2:
            return [None] * len(input_ids)

        approximated: list[Optional[int]] = [None]
        approximated.extend(range(min(word_count, len(input_ids) - 2)))
        approximated.extend([None] * (len(input_ids) - len(approximated)))
        return approximated[: len(input_ids)]

    def _tokenize_words(self, tokenizer: Any, text: str) -> tuple[Any, list[str]]:
        words = text.split()
        try:
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        except (TypeError, ValueError):
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        return encoding, words

    def _attention_importance(self, text: str, rng: random.Random) -> list[float]:
        """Compute legacy word importance using masked-LM attention weights."""
        if not self._ensure_fallback_model() or self._model is None:
            return [rng.random() for _ in text.split()]

        import torch

        encoding, words = self._tokenize_words(self._tokenizer, text)
        tokens = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**tokens, output_attentions=True)

        attentions = outputs.attentions
        if not attentions:
            return [rng.random() for _ in words]

        avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))
        if avg_attention.dim() > 1:
            avg_attention = avg_attention.mean(dim=tuple(range(avg_attention.dim() - 1)))

        word_ids = self._word_ids(encoding, len(words))
        word_scores = [0.0] * len(words)
        word_counts = [0] * len(words)
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None and word_index < len(words):
                word_scores[word_index] += float(avg_attention[token_index].item())
                word_counts[word_index] += 1

        for index, count in enumerate(word_counts):
            if count:
                word_scores[index] /= count
        return word_scores

    def _get_token_importance(self, text: str) -> list[float]:
        """Return legacy attention importance scores for backward compatibility."""
        return self._attention_importance(text, self._rng_for(text, self.strength))

    def _adversarial_replace(self, text: str, token_indices: list[int]) -> str:
        """Run the legacy MLM replacement path for backward compatibility."""
        return self._attention_replace(
            text,
            token_indices,
            self._rng_for(text, self.strength),
        )

    def _gradient_analysis(self, text: str) -> _GradientAnalysis:
        """Compute ``grad(loss, embeddings).norm(dim=-1)`` and word gradients."""
        if not self.gradient_available:
            raise RuntimeError("A target model and tokenizer are required for gradient attacks")

        import torch
        import torch.nn.functional as F  # noqa: N812

        model = self._unwrap_model(self.target_model)
        tokenizer = self._gradient_tokenizer
        if not hasattr(model, "get_input_embeddings"):
            raise TypeError("target_model must expose get_input_embeddings()")

        embedding_layer = model.get_input_embeddings()
        if embedding_layer is None:
            raise TypeError("target_model returned no input embedding layer")

        encoding, words = self._tokenize_words(tokenizer, text)
        device = embedding_layer.weight.device
        model_inputs = {
            key: value.to(device) for key, value in encoding.items() if key != "input_ids"
        }
        input_ids = encoding["input_ids"].to(device)
        embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.enable_grad():
                outputs = model(inputs_embeds=embeddings, **model_inputs)
                logits = outputs.logits
                if logits.dim() == 2:
                    labels = logits.detach().argmax(dim=-1)
                    loss = F.cross_entropy(logits, labels)
                elif logits.dim() == 3:
                    vocabulary_sized = int(input_ids.max().item()) < logits.shape[-1]
                    model_class = type(model).__name__.casefold()
                    model_config = getattr(model, "config", None)
                    is_causal = (
                        "causallm" in model_class
                        or "gpt" in model_class
                        or bool(getattr(model_config, "is_decoder", False))
                    )
                    if vocabulary_sized and is_causal and logits.shape[1] > 1:
                        loss = F.cross_entropy(
                            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )
                    else:
                        if vocabulary_sized:
                            labels = input_ids
                        else:
                            labels = logits.detach().argmax(dim=-1)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            labels.reshape(-1),
                        )
                else:
                    raise ValueError("target_model logits must be rank 2 or 3 for learned attacks")

                gradients = torch.autograd.grad(
                    loss,
                    embeddings,
                    create_graph=False,
                    retain_graph=False,
                )[0][0]
        finally:
            if was_training:
                model.train()

        token_scores = gradients.norm(dim=-1)
        word_ids = self._word_ids(encoding, len(words))
        word_scores = [0.0] * len(words)
        word_gradients: list[Any] = [None] * len(words)
        word_token_ids: list[list[int]] = [[] for _ in words]
        counts = [0] * len(words)

        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= len(words):
                continue
            word_scores[word_index] += float(token_scores[token_index].item())
            gradient = gradients[token_index].detach()
            if word_gradients[word_index] is None:
                word_gradients[word_index] = gradient.clone()
            else:
                word_gradients[word_index] += gradient
            word_token_ids[word_index].append(int(input_ids[0, token_index].item()))
            counts[word_index] += 1

        zero_gradient = torch.zeros(
            embedding_layer.embedding_dim,
            device=device,
            dtype=embedding_layer.weight.dtype,
        )
        for index, count in enumerate(counts):
            if count:
                word_scores[index] /= count
                word_gradients[index] /= count
            else:
                word_gradients[index] = zero_gradient.clone()

        return _GradientAnalysis(word_scores, word_gradients, word_token_ids)

    @staticmethod
    def _normalise_candidate(candidate: str, original: str) -> Optional[str]:
        candidate = candidate.strip().removeprefix("##")
        candidate = candidate.lstrip("Ġ▁")
        if not candidate or len(candidate.split()) != 1:
            return None
        if not any(character.isalnum() for character in candidate):
            return None
        if candidate.casefold() == original.casefold():
            return None
        if original[:1].isupper():
            candidate = candidate[:1].upper() + candidate[1:]
        return candidate

    def _gradient_replace(
        self,
        text: str,
        token_indices: list[int],
        analysis: _GradientAnalysis,
    ) -> str:
        """Select substitutions by projecting gradients onto the embedding matrix."""
        import torch

        model = self._unwrap_model(self.target_model)
        tokenizer = self._gradient_tokenizer
        if tokenizer is None:
            raise RuntimeError("A target tokenizer is required for gradient substitutions")
        embedding_matrix = model.get_input_embeddings().weight.detach()
        words = text.split()
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        for word_index in token_indices:
            if word_index >= len(words):
                continue
            gradient = analysis.gradients[word_index].to(embedding_matrix.device)
            if not torch.isfinite(gradient).all() or float(gradient.norm().item()) == 0.0:
                continue

            scores = torch.mv(embedding_matrix, gradient)
            original_token_ids = set(analysis.token_ids[word_index])
            blocked_ids = special_ids | original_token_ids
            if blocked_ids:
                blocked_tensor = torch.tensor(
                    sorted(blocked_ids),
                    device=scores.device,
                    dtype=torch.long,
                )
                valid_blocked = blocked_tensor[blocked_tensor < scores.numel()]
                scores[valid_blocked] = -torch.inf

            top_count = min(64, scores.numel())
            candidate_ids = torch.topk(scores, k=top_count).indices.tolist()
            for candidate_id in candidate_ids:
                candidate = tokenizer.decode(
                    [candidate_id],
                    skip_special_tokens=True,
                )
                normalised = self._normalise_candidate(candidate, words[word_index])
                if normalised is not None:
                    words[word_index] = normalised
                    break

        return " ".join(words)

    def _attention_replace(
        self,
        text: str,
        token_indices: list[int],
        rng: random.Random,
    ) -> str:
        words = text.split()
        if not self._ensure_fallback_model() or self._model is None:
            fallback_words = [
                "however",
                "never",
                "always",
                "perhaps",
                "indeed",
                "actually",
                "certainly",
                "rarely",
                "frequently",
                "surprisingly",
            ]
            for index in token_indices:
                if 0 <= index < len(words):
                    words[index] = rng.choice(fallback_words)
            return " ".join(words)

        import torch

        tokenizer = self._tokenizer
        if tokenizer is None:
            logger.debug("Fallback tokenizer unavailable; returning the original text.")
            return text

        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id
        if mask_token is None or mask_token_id is None:
            logger.debug("Fallback tokenizer has no mask token; returning the original text.")
            return text

        for index in token_indices:
            if not 0 <= index < len(words):
                continue

            masked_words = list(words)
            masked_words[index] = mask_token
            encoding = tokenizer(
                " ".join(masked_words),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            tokens = {key: value.to(self.device) for key, value in encoding.items()}
            with torch.no_grad():
                outputs = self._model(**tokens)

            mask_positions = (tokens["input_ids"] == mask_token_id).nonzero(as_tuple=True)
            if len(mask_positions[1]) == 0:
                continue

            logits = outputs.logits[0, mask_positions[1][0].item()]
            candidate_ids = logits.topk(min(10, logits.numel())).indices.tolist()
            for candidate_id in candidate_ids:
                candidate = tokenizer.decode(
                    [candidate_id],
                    skip_special_tokens=True,
                )
                normalised = self._normalise_candidate(
                    candidate,
                    words[index],
                )
                if normalised is not None:
                    words[index] = normalised
                    break

        return " ".join(words)

    def _cache_key(self, text: str, strength: float, strategy: str) -> tuple:
        model_id = id(self.target_model) if self.target_model else 0
        return (self.cycle_id, model_id, text, round(float(strength), 8), strategy)

    def _cache_result(self, key: tuple, value: str) -> None:
        if not self.cache_enabled:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)

    def generate(
        self,
        text: str,
        strength: Optional[float] = None,
        cycle_id: Optional[int] = None,
    ) -> str:
        """Generate an adversarial example for ``text``.

        Args:
            text: Input text.
            strength: Optional distortion strength override.
            cycle_id: Optional cycle identifier for cache partitioning.

        Returns:
            Adversarially modified text.
        """
        if not text or not text.strip():
            return text

        actual_strength = self.strength if strength is None else float(strength)
        if actual_strength <= 0.0:
            return text
        actual_strength = min(actual_strength, 1.0)
        if cycle_id is not None:
            self.set_cycle(cycle_id)

        words = text.split()
        if len(words) < 2:
            return text

        effective_strategy = self.strategy
        if effective_strategy == "gradient" and not self.gradient_available:
            logger.info(
                "Gradient learned distortion requested without a target model/tokenizer; "
                "falling back to attention strategy."
            )
            effective_strategy = "attention"

        key = self._cache_key(text, actual_strength, effective_strategy)
        if self.cache_enabled and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        num_to_replace = max(1, int(len(words) * actual_strength * 0.4))
        rng = self._rng_for(text, actual_strength)

        if effective_strategy == "gradient":
            try:
                analysis = self._gradient_analysis(text)
                ranked = sorted(
                    enumerate(analysis.scores),
                    key=lambda item: item[1],
                    reverse=True,
                )
                target_indices = [index for index, _ in ranked[:num_to_replace]]
                result = self._gradient_replace(text, target_indices, analysis)
            except Exception:
                logger.warning(
                    "Gradient learned distortion failed; using attention fallback.",
                    exc_info=True,
                )
                importance = self._attention_importance(text, rng)
                ranked = sorted(
                    enumerate(importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
                target_indices = [index for index, _ in ranked[:num_to_replace]]
                result = self._attention_replace(text, target_indices, rng)
                fallback_key = self._cache_key(text, actual_strength, "attention_fallback")
                self._cache_result(fallback_key, result)
                return result
        else:
            importance = self._attention_importance(text, rng)
            ranked = sorted(
                enumerate(importance),
                key=lambda item: item[1],
                reverse=True,
            )
            target_indices = [index for index, _ in ranked[:num_to_replace]]
            result = self._attention_replace(text, target_indices, rng)

        self._cache_result(key, result)
        return result
