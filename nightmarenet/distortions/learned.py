"""Learned adversarial distortions using a masked language model.

Uses a small MLM (e.g. distilbert-base-uncased) to generate adversarial
token replacements that maximize confusion. Falls back to random replacement
if the model is unavailable.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LearnedAdversarialGenerator:
    """Generates adversarial text by replacing important tokens with MLM predictions.

    Args:
        model_name: Name of the masked LM to use for adversarial replacements.
        device: Device to run the MLM on.
        strength: Default distortion strength (0-1).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cpu",
        strength: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device
        self.strength = strength
        self._model: Any = None
        self._tokenizer: Any = None
        self._available = True

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name)
            self._model.to(device)
            self._model.eval()
            logger.info("Loaded learned adversarial model: %s", model_name)
        except Exception as exc:
            logger.warning(
                "Could not load adversarial model '%s': %s. "
                "Falling back to random replacement.",
                model_name,
                exc,
            )
            self._available = False

    def _get_token_importance(self, text: str) -> list[float]:
        """Compute per-token importance scores using attention weights.

        Args:
            text: Input text.

        Returns:
            List of importance scores (one per word-level token).
        """
        if not self._available or self._model is None:
            words = text.split()
            return [random.random() for _ in words]

        import torch

        encoding = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**tokens, output_attentions=True)

        # Average attention across all heads and layers
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))  # (seq_len,)

        # Map subword attention back to word-level
        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None
        if word_ids is None:
            # Fall back: use subword scores directly, one per word
            scores = avg_attention[1:-1].cpu().tolist()  # skip [CLS] and [SEP]
            words = text.split()
            if len(scores) >= len(words):
                return scores[: len(words)]
            return scores + [0.0] * (len(words) - len(scores))

        # Aggregate subword attention to word level
        words = text.split()
        word_scores = [0.0] * len(words)
        word_counts = [0] * len(words)
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(words):
                word_scores[wid] += avg_attention[i].item()
                word_counts[wid] += 1
        for i in range(len(words)):
            if word_counts[i] > 0:
                word_scores[i] /= word_counts[i]

        return word_scores

    def _adversarial_replace(self, text: str, token_indices: list[int]) -> str:
        """Replace tokens at given indices with MLM-predicted confusing alternatives.

        Args:
            text: Input text.
            token_indices: Indices of words to replace.

        Returns:
            Text with adversarial replacements.
        """
        words = text.split()
        if not self._available or self._model is None:
            # Fallback: replace with random common words
            fallback_words = [
                "however", "never", "always", "perhaps", "indeed",
                "actually", "certainly", "rarely", "frequently", "surprisingly",
            ]
            for idx in token_indices:
                if 0 <= idx < len(words):
                    words[idx] = random.choice(fallback_words)
            return " ".join(words)

        import torch

        mask_token = self._tokenizer.mask_token
        for idx in token_indices:
            if 0 <= idx < len(words):
                masked_words = list(words)
                masked_words[idx] = mask_token
                masked_text = " ".join(masked_words)

                tokens = self._tokenizer(
                    masked_text, return_tensors="pt", truncation=True, max_length=512
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = self._model(**tokens)

                # Find the mask position
                mask_token_id = self._tokenizer.mask_token_id
                mask_positions = (tokens["input_ids"] == mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[1]) == 0:
                    continue

                mask_pos = mask_positions[1][0].item()
                logits = outputs.logits[0, mask_pos]

                # Pick a prediction that differs from the original word
                # Use top-k and pick a non-original one to maximize confusion
                top_k = 10
                top_indices = logits.topk(top_k).indices.tolist()
                original_word = words[idx].lower()

                replacement = None
                for tid in top_indices:
                    candidate = self._tokenizer.decode([tid]).strip()
                    if candidate.lower() != original_word and len(candidate) > 1:
                        replacement = candidate
                        break

                if replacement:
                    words[idx] = replacement

        return " ".join(words)

    def generate(self, text: str, strength: Optional[float] = None) -> str:
        """Generate adversarial text by replacing high-importance tokens.

        Args:
            text: Input text.
            strength: Distortion strength 0-1. Higher = more tokens replaced.

        Returns:
            Adversarially modified text.
        """
        if not text or not text.strip():
            return text

        strength = strength if strength is not None else self.strength
        words = text.split()
        if len(words) < 2:
            return text

        # Get importance scores
        importance = self._get_token_importance(text)

        # Select top-N important tokens to replace based on strength
        num_to_replace = max(1, int(len(words) * strength * 0.4))

        # Sort indices by importance (descending) and take top ones
        indexed_scores = list(enumerate(importance))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        target_indices = [idx for idx, _ in indexed_scores[:num_to_replace]]

        return self._adversarial_replace(text, target_indices)
