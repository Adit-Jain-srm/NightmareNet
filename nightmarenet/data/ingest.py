"""Unified data ingestion: URLs, files, or HuggingFace Hub.

Provides a single ``DataIngestor`` that normalises any input source
into a ``datasets.Dataset`` with a ``text`` column ready for the
NightmareNet training pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Optional

from datasets import Dataset

from nightmarenet.data.loader import DatasetWrapper
from nightmarenet.data.scraper import WebScraper

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 10  # absolute minimum for a usable corpus


class DataIngestor:
    """Normalise heterogeneous data sources into a HuggingFace Dataset.

    Supported sources:

    * ``urls``  — list of web-page URLs → scrape → Dataset
    * ``file``  — path to ``.txt``, ``.csv``, or ``.jsonl`` file
    * ``huggingface`` — HuggingFace Hub dataset name (+ optional subset)

    Args:
        text_column: Name of the text column in the output Dataset.
        max_samples: Optional cap on the number of samples.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.text_column = text_column
        self.max_samples = max_samples
        self.seed = seed

    # ------------------------------------------------------------------
    # URL ingestion
    # ------------------------------------------------------------------

    def from_urls(
        self,
        urls: list[str],
        *,
        delay: float = 1.0,
        chunk_size: int = 512,
    ) -> Dataset:
        """Scrape *urls* and return a Dataset.

        Args:
            urls: List of page URLs to scrape.
            delay: Seconds between requests.
            chunk_size: Target characters per sample chunk.

        Returns:
            Dataset with a ``text`` column.
        """
        scraper = WebScraper(delay=delay)
        dataset = scraper.scrape(urls, chunk_size=chunk_size)
        return self._finalise(dataset, f"urls({len(urls)})")

    # ------------------------------------------------------------------
    # File ingestion
    # ------------------------------------------------------------------

    def from_file(self, path: str) -> Dataset:
        """Load a local file into a Dataset.

        Supported formats:

        * ``.txt`` — one sample per paragraph (blank-line separated)
        * ``.csv`` — expects a column named after ``self.text_column``
        * ``.jsonl`` — expects a key named after ``self.text_column``

        Args:
            path: Absolute or relative path to the file.

        Returns:
            Dataset with a ``text`` column.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".txt":
            dataset = self._load_txt(path)
        elif ext == ".csv":
            dataset = self._load_csv(path)
        elif ext == ".jsonl":
            dataset = self._load_jsonl(path)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: .txt, .csv, .jsonl")

        return self._finalise(dataset, f"file({os.path.basename(path)})")

    def from_text_content(self, content: str, source_name: str = "upload") -> Dataset:
        """Create a Dataset from raw text content (e.g. from an upload).

        Splits on double-newlines (paragraphs).

        Args:
            content: Raw text string.
            source_name: Label for logging.

        Returns:
            Dataset with a ``text`` column.
        """
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            # Fall back to single-newline splitting
            paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 20]
        if not paragraphs:
            raise ValueError("Could not extract any text samples from content.")
        dataset = Dataset.from_dict({self.text_column: paragraphs})
        return self._finalise(dataset, source_name)

    # ------------------------------------------------------------------
    # HuggingFace Hub ingestion
    # ------------------------------------------------------------------

    def from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        streaming: bool = False,
    ) -> Dataset:
        """Load a dataset from HuggingFace Hub.

        Uses the existing ``DatasetWrapper`` for full compatibility with
        the rest of the NightmareNet training pipeline.

        Args:
            dataset_name: HuggingFace dataset name (e.g. ``"wikitext"``).
            subset: Optional subset (e.g. ``"wikitext-2-raw-v1"``).
            streaming: Whether to use streaming mode.

        Returns:
            Dataset with a ``text`` column.
        """
        wrapper = DatasetWrapper(
            dataset_name=dataset_name,
            subset=subset,
            text_column=self.text_column,
            max_samples=self.max_samples,
            seed=self.seed,
            streaming=streaming,
        ).load()
        if streaming:
            return wrapper.train_data
        return self._finalise(wrapper.train_data, f"huggingface({dataset_name})")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_txt(self, path: str) -> Dataset:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        if len(paragraphs) <= 1:
            paragraphs = [ln.strip() for ln in content.splitlines() if len(ln.strip()) > 20]
        return Dataset.from_dict({self.text_column: paragraphs})

    def _load_csv(self, path: str) -> Dataset:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or self.text_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV file must contain a '{self.text_column}' column. "
                    f"Found: {reader.fieldnames}"
                )
            texts = [row[self.text_column] for row in reader if row.get(self.text_column)]
        return Dataset.from_dict({self.text_column: texts})

    def _load_jsonl(self, path: str) -> Dataset:
        texts: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line %d: %s", line_no, exc)
                    continue
                if self.text_column in obj and obj[self.text_column]:
                    texts.append(str(obj[self.text_column]))
        if not texts:
            raise ValueError(f"No valid entries with key '{self.text_column}' found in JSONL file.")
        return Dataset.from_dict({self.text_column: texts})

    def _finalise(self, dataset: Dataset, source_label: str) -> Dataset:
        """Filter empties, limit samples, and log summary."""
        # Filter empty / whitespace-only texts
        dataset = dataset.filter(
            lambda x: bool(x[self.text_column] and x[self.text_column].strip())
        )

        if len(dataset) < _MIN_SAMPLES:
            raise ValueError(
                f"Source '{source_label}' produced only {len(dataset)} usable samples "
                f"(minimum {_MIN_SAMPLES}). Provide more data."
            )

        if self.max_samples is not None and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))

        logger.info(
            "Ingested %d samples from %s.",
            len(dataset),
            source_label,
        )
        return dataset
