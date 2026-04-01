"""Dream and nightmare data generation.

Applies distortions to a base dataset to produce dream (mildly distorted)
and nightmare (extremely perturbed) training splits.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from datasets import Dataset

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.utils.validation import (
    validate_dataset_columns,
    validate_non_empty_dataset,
    validate_strength,
)

logger = logging.getLogger(__name__)


class DreamDatasetGenerator:
    """Generates mildly distorted dream data from a base dataset.

    Applies text-level and light semantic distortions to encourage
    the model to learn abstract, invariant representations.

    Args:
        strength: Distortion strength (0–1). Typical dream range: 0.2–0.3.
        text_column: Name of the text column in the dataset.
        config: Optional distortion config dict with per-type weights.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        strength: float = 0.25,
        text_column: str = "text",
        config: Optional[dict] = None,
        seed: int = 42,
    ):
        self.strength = validate_strength(strength, "strength")
        self.text_column = text_column
        self.config = config or {}
        self.seed = seed

    def _distort(self, example: dict) -> dict:
        """Apply dream-level distortions to a single example."""
        text = example[self.text_column]
        if not text or not text.strip():
            return example

        # Apply text-level corruptions (primary for dream phase)
        text_config = self.config.get("text", None)
        result = apply_text_distortions(text, strength=self.strength, config=text_config)

        # Apply light semantic distortions
        semantic_config = self.config.get("semantic", None)
        result = apply_semantic_distortions(
            result, strength=self.strength * 0.5, config=semantic_config
        )

        return {**example, self.text_column: result}

    def generate(self, dataset: Dataset) -> Dataset:
        """Generate a dream dataset by applying mild distortions.

        Args:
            dataset: Base HuggingFace Dataset to distort.

        Returns:
            A new Dataset with mildly distorted text.
        """
        import random

        random.seed(self.seed)

        validate_dataset_columns(dataset, [self.text_column])
        validate_non_empty_dataset(dataset, "dataset")

        logger.info(
            "Generating dream data (strength=%.2f) from %d samples...",
            self.strength,
            len(dataset),
        )

        original_texts = dataset[self.text_column]

        dream_data = dataset.map(
            self._distort,
            desc="Generating dream data",
        )

        modified_count = sum(
            1 for o, g in zip(original_texts, dream_data[self.text_column]) if o != g
        )
        logger.info(
            "Dream data generation complete. %d samples produced, %d texts modified.",
            len(dream_data),
            modified_count,
        )
        return dream_data

    def generate_and_save(self, dataset: Dataset, output_dir: str) -> Dataset:
        """Generate dream data and save to disk.

        Args:
            dataset: Base dataset to distort.
            output_dir: Directory to save the generated dataset.

        Returns:
            The generated dream Dataset.
        """
        dream_data = self.generate(dataset)
        save_path = os.path.join(output_dir, "dream")
        try:
            os.makedirs(save_path, exist_ok=True)
            dream_data.save_to_disk(save_path)
        except OSError as exc:
            raise OSError(
                f"Failed to save dream data to '{save_path}': {exc}"
            ) from exc
        logger.info("Dream data saved to %s", save_path)
        return dream_data


class NightmareDatasetGenerator:
    """Generates extremely perturbed nightmare data from a base dataset.

    Applies aggressive text, semantic, and adversarial distortions to
    stress-test the model's learned representations.

    Args:
        strength: Distortion strength (0–1). Typical nightmare range: 0.7–0.9.
        text_column: Name of the text column in the dataset.
        config: Optional distortion config dict with per-type weights.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        strength: float = 0.8,
        text_column: str = "text",
        config: Optional[dict] = None,
        seed: int = 42,
    ):
        self.strength = validate_strength(strength, "strength")
        self.text_column = text_column
        self.config = config or {}
        self.seed = seed

    def _distort(self, example: dict) -> dict:
        """Apply nightmare-level distortions to a single example."""
        text = example[self.text_column]
        if not text or not text.strip():
            return example

        # Apply aggressive text-level corruptions
        text_config = self.config.get("text", None)
        result = apply_text_distortions(text, strength=self.strength, config=text_config)

        # Apply strong semantic distortions
        semantic_config = self.config.get("semantic", None)
        result = apply_semantic_distortions(
            result, strength=self.strength, config=semantic_config
        )

        # Apply adversarial distortions (unique to nightmare phase)
        adversarial_config = self.config.get("adversarial", None)
        result = apply_adversarial_distortions(
            result, strength=self.strength, config=adversarial_config
        )

        return {**example, self.text_column: result}

    def generate(self, dataset: Dataset) -> Dataset:
        """Generate a nightmare dataset by applying extreme distortions.

        Args:
            dataset: Base HuggingFace Dataset to distort.

        Returns:
            A new Dataset with extremely perturbed text.
        """
        import random

        random.seed(self.seed)

        validate_dataset_columns(dataset, [self.text_column])
        validate_non_empty_dataset(dataset, "dataset")

        logger.info(
            "Generating nightmare data (strength=%.2f) from %d samples...",
            self.strength,
            len(dataset),
        )

        original_texts = dataset[self.text_column]

        nightmare_data = dataset.map(
            self._distort,
            desc="Generating nightmare data",
        )

        modified_count = sum(
            1 for o, g in zip(original_texts, nightmare_data[self.text_column]) if o != g
        )
        logger.info(
            "Nightmare data generation complete. %d samples produced, %d texts modified.",
            len(nightmare_data),
            modified_count,
        )
        return nightmare_data

    def generate_and_save(self, dataset: Dataset, output_dir: str) -> Dataset:
        """Generate nightmare data and save to disk.

        Args:
            dataset: Base dataset to distort.
            output_dir: Directory to save the generated dataset.

        Returns:
            The generated nightmare Dataset.
        """
        nightmare_data = self.generate(dataset)
        save_path = os.path.join(output_dir, "nightmare")
        try:
            os.makedirs(save_path, exist_ok=True)
            nightmare_data.save_to_disk(save_path)
        except OSError as exc:
            raise OSError(
                f"Failed to save nightmare data to '{save_path}': {exc}"
            ) from exc
        logger.info("Nightmare data saved to %s", save_path)
        return nightmare_data


def create_generators_from_config(config: dict) -> tuple[DreamDatasetGenerator, NightmareDatasetGenerator]:
    """Create dream and nightmare generators from a config dictionary.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (DreamDatasetGenerator, NightmareDatasetGenerator).
    """
    distortion_config = config.get("distortion", {})
    dataset_config = config.get("dataset", {})
    seed = config.get("seed", 42)

    dream_gen = DreamDatasetGenerator(
        strength=distortion_config.get("dream_strength", 0.25),
        text_column=dataset_config.get("text_column", "text"),
        config=distortion_config,
        seed=seed,
    )

    nightmare_gen = NightmareDatasetGenerator(
        strength=distortion_config.get("nightmare_strength", 0.8),
        text_column=dataset_config.get("text_column", "text"),
        config=distortion_config,
        seed=seed,
    )

    return dream_gen, nightmare_gen
