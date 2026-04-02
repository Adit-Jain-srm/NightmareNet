"""Dataset loading utilities.

Wraps HuggingFace `datasets` to provide a unified interface for loading text
datasets and returning raw, dream, and nightmare splits.
"""

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, load_dataset

from nightmarenet.utils.validation import (
    validate_dataset_columns,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class DatasetWrapper:
    """Unified wrapper around HuggingFace datasets.

    Loads a text dataset and provides access to raw data, with methods to
    generate dream and nightmare splits via the generator module.

    Args:
        dataset_name: Name of the HuggingFace dataset (e.g., "wikitext").
        subset: Optional dataset subset (e.g., "wikitext-2-raw-v1").
        text_column: Name of the column containing text data.
        max_samples: Optional limit on the number of samples to load.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.text_column = text_column
        self.max_samples = max_samples
        self.seed = seed

        self._train_dataset = None
        self._test_dataset = None

    def load(self) -> DatasetWrapper:
        """Load the dataset from HuggingFace Hub.

        Returns:
            self for chaining.
        """
        if self.max_samples is not None:
            validate_positive_int(self.max_samples, "max_samples")

        logger.info(
            "Loading dataset '%s' (subset=%s)", self.dataset_name, self.subset
        )

        kwargs = {"path": self.dataset_name}
        if self.subset:
            kwargs["name"] = self.subset

        try:
            raw = load_dataset(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{self.dataset_name}' "
                f"(subset={self.subset}): {exc}"
            ) from exc

        # Get train and test splits
        if "train" in raw:
            self._train_dataset = raw["train"]
        elif "all" in raw:
            self._train_dataset = raw["all"]
        else:
            # Use the first available split
            first_split = list(raw.keys())[0]
            self._train_dataset = raw[first_split]
            logger.warning(
                "No 'train' split found; using '%s' split.", first_split
            )

        if "test" in raw:
            self._test_dataset = raw["test"]
        elif "validation" in raw:
            self._test_dataset = raw["validation"]
        else:
            # Split train data
            split = self._train_dataset.train_test_split(
                test_size=0.1, seed=self.seed
            )
            self._train_dataset = split["train"]
            self._test_dataset = split["test"]
            logger.info("Created test split from training data (10%%).")

        # Validate that the text column exists
        validate_dataset_columns(self._train_dataset, [self.text_column])
        validate_dataset_columns(self._test_dataset, [self.text_column])

        # Filter out empty texts
        train_count_before = len(self._train_dataset)
        test_count_before = len(self._test_dataset)
        self._train_dataset = self._train_dataset.filter(
            lambda x: bool(x[self.text_column] and x[self.text_column].strip())
        )
        self._test_dataset = self._test_dataset.filter(
            lambda x: bool(x[self.text_column] and x[self.text_column].strip())
        )

        train_filtered = train_count_before - len(self._train_dataset)
        test_filtered = test_count_before - len(self._test_dataset)
        if train_count_before > 0 and train_filtered > train_count_before * 0.5:
            logger.warning(
                "More than 50%% of training data was filtered as empty "
                "(%d of %d samples removed).",
                train_filtered,
                train_count_before,
            )
        if test_count_before > 0 and test_filtered > test_count_before * 0.5:
            logger.warning(
                "More than 50%% of test data was filtered as empty "
                "(%d of %d samples removed).",
                test_filtered,
                test_count_before,
            )

        # Limit samples if requested
        if self.max_samples is not None:
            if len(self._train_dataset) > self.max_samples:
                self._train_dataset = self._train_dataset.select(
                    range(self.max_samples)
                )
                logger.info("Limited training data to %d samples.", self.max_samples)
            test_limit = min(len(self._test_dataset), self.max_samples // 5)
            if test_limit > 0:
                self._test_dataset = self._test_dataset.select(range(test_limit))

        logger.info(
            "Loaded %d train samples, %d test samples.",
            len(self._train_dataset),
            len(self._test_dataset),
        )
        return self

    @property
    def train_data(self) -> Dataset:
        """Return the training dataset."""
        if self._train_dataset is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._train_dataset

    @property
    def test_data(self) -> Dataset:
        """Return the test dataset."""
        if self._test_dataset is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._test_dataset

    def get_texts(self, split: str = "train") -> list[str]:
        """Return a list of text strings from the specified split.

        Args:
            split: "train" or "test".

        Returns:
            List of text strings.
        """
        dataset = self.train_data if split == "train" else self.test_data
        return dataset[self.text_column]


def load_from_config(config: dict) -> DatasetWrapper:
    """Create and load a DatasetWrapper from a config dictionary.

    Args:
        config: Configuration dictionary with 'dataset' and 'seed' keys.

    Returns:
        Loaded DatasetWrapper instance.
    """
    dataset_config = config.get("dataset", {})
    return DatasetWrapper(
        dataset_name=dataset_config.get("name", "wikitext"),
        subset=dataset_config.get("subset", "wikitext-2-raw-v1"),
        text_column=dataset_config.get("text_column", "text"),
        max_samples=dataset_config.get("max_samples"),
        seed=config.get("seed", 42),
    ).load()
