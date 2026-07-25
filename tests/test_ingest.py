"""Tests for the DataIngestor module."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from nightmarenet.data.ingest import DataIngestor


@pytest.fixture
def ingestor():
    return DataIngestor(text_column="text", max_samples=None, seed=42)


# ── TXT ingestion ──


class TestTxtIngestion:
    def test_load_txt_paragraphs(self, ingestor):
        """Paragraphs separated by blank lines become samples."""
        content = "\n\n".join(
            [
                f"This is paragraph number {i} with enough text to pass the filter threshold."
                for i in range(20)
            ]
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            path = f.name

        try:
            ds = ingestor.from_file(path)
            assert "text" in ds.column_names
            assert len(ds) >= 10
        finally:
            os.unlink(path)

    def test_load_txt_fallback_to_lines(self, ingestor):
        """When no blank-line paragraphs, fall back to single-line splitting."""
        content = "\n".join(
            [
                f"Line {i}: This contains enough text to be a valid training sample."
                for i in range(20)
            ]
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            path = f.name

        try:
            ds = ingestor.from_file(path)
            assert len(ds) >= 10
        finally:
            os.unlink(path)


# ── CSV ingestion ──


class TestCsvIngestion:
    def test_load_csv(self, ingestor):
        """CSV with a text column should be loaded."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("text,label\n")
            for i in range(20):
                f.write(f'"This is sample {i} with meaningful content for training.",{i}\n')
            path = f.name

        try:
            ds = ingestor.from_file(path)
            assert "text" in ds.column_names
            assert len(ds) >= 10
        finally:
            os.unlink(path)

    def test_csv_missing_column_raises(self, ingestor):
        """CSV without the expected text column should raise."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("content,label\n")
            f.write("hello,1\n")
            path = f.name

        try:
            with pytest.raises(ValueError, match="text"):
                ingestor.from_file(path)
        finally:
            os.unlink(path)


# ── JSONL ingestion ──


class TestJsonlIngestion:
    def test_load_jsonl(self, ingestor):
        """JSONL with a text key should be loaded."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Sample {i}: enough text for training."}) + "\n")
            path = f.name

        try:
            ds = ingestor.from_file(path)
            assert len(ds) >= 10
        finally:
            os.unlink(path)

    def test_jsonl_skips_malformed(self, ingestor):
        """Malformed JSONL lines should be skipped without crashing."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for i in range(15):
                f.write(json.dumps({"text": f"Valid sample {i} with real content."}) + "\n")
            f.write("this is not json\n")
            f.write('{"text": "Another valid sample at the end."}\n')
            path = f.name

        try:
            ds = ingestor.from_file(path)
            assert len(ds) >= 10
        finally:
            os.unlink(path)


# ── Text content ingestion ──


class TestTextContent:
    def test_from_text_content(self, ingestor):
        """Raw text should be split into paragraphs."""
        content = "\n\n".join(
            [
                f"Paragraph {i}: This is a valid paragraph with substantial training text."
                for i in range(15)
            ]
        )
        ds = ingestor.from_text_content(content)
        assert len(ds) >= 10

    def test_empty_content_raises(self, ingestor):
        """Empty text content should raise."""
        with pytest.raises(ValueError):
            ingestor.from_text_content("")


# ── Edge cases ──


class TestEdgeCases:
    def test_unsupported_extension_raises(self, ingestor):
        """Unsupported file types should raise ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            delete=False,
        ) as f:
            f.write("<root>hello</root>")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                ingestor.from_file(path)
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self, ingestor):
        """Non-existent file paths should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ingestor.from_file("/nonexistent/path/data.txt")

    def test_max_samples_caps_output(self):
        """max_samples should limit the output dataset size."""
        ingestor = DataIngestor(max_samples=15)
        content = "\n\n".join(
            [
                f"Paragraph {i}: This paragraph has plenty of content for model training purposes."
                for i in range(50)
            ]
        )
        ds = ingestor.from_text_content(content)
        assert len(ds) <= 15

    def test_minimum_samples_threshold_raises(self, ingestor):
        """_finalise should raise ValueError if resulting dataset has fewer than _MIN_SAMPLES."""
        # We provide only 5 samples (less than 10)
        content = "\n\n".join(
            [f"Paragraph {i}: This is valid but not enough samples." for i in range(5)]
        )
        with pytest.raises(ValueError, match="produced only 5 usable samples"):
            ingestor.from_text_content(content)

    def test_minimum_samples_threshold_exact(self, ingestor):
        """_finalise should NOT raise ValueError if resulting dataset has exactly _MIN_SAMPLES."""
        content = "\n\n".join(
            [
                f"Paragraph {i}: This is valid and exactly enough samples to pass the threshold."
                for i in range(10)
            ]
        )
        ds = ingestor.from_text_content(content)
        assert len(ds) == 10


# ── URL ingestion ──


class TestUrlIngestion:
    @patch("nightmarenet.data.ingest.WebScraper")
    def test_from_urls(self, mock_scraper_class, ingestor):
        """WebScraper should be instantiated and used to process URLs."""
        mock_scraper_instance = mock_scraper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict({"text": [f"Valid sample {i} content" for i in range(15)]})
        mock_scraper_instance.scrape.return_value = mock_ds

        urls = ["http://example.com/1", "http://example.com/2"]
        ds = ingestor.from_urls(urls, delay=0.5, chunk_size=256)

        mock_scraper_class.assert_called_once_with(delay=0.5)
        mock_scraper_instance.scrape.assert_called_once_with(urls, chunk_size=256)
        assert len(ds) == 15
        assert "text" in ds.column_names

    @patch("nightmarenet.data.ingest.WebScraper")
    def test_from_urls_below_threshold_raises(self, mock_scraper_class, ingestor):
        """from_urls should raise ValueError if scraper returns < 10 samples."""
        mock_scraper_instance = mock_scraper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict({"text": [f"Valid sample {i} content" for i in range(5)]})
        mock_scraper_instance.scrape.return_value = mock_ds

        urls = ["http://example.com/1"]
        with pytest.raises(ValueError, match="produced only 5 usable samples"):
            ingestor.from_urls(urls)


# ── HuggingFace Hub ingestion ──


class TestHuggingFaceIngestion:
    @patch("nightmarenet.data.ingest.DatasetWrapper")
    def test_from_huggingface(self, mock_wrapper_class, ingestor):
        """DatasetWrapper should be used to load HuggingFace datasets."""
        mock_wrapper_instance = mock_wrapper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict({"text": [f"HF sample {i} content" for i in range(15)]})
        mock_wrapper_instance.train_data = mock_ds
        mock_wrapper_instance.load.return_value = mock_wrapper_instance

        ds = ingestor.from_huggingface("glue", subset="sst2", streaming=True)

        mock_wrapper_class.assert_called_once_with(
            dataset_name="glue",
            subset="sst2",
            text_column="text",
            max_samples=None,
            seed=42,
            streaming=True,
        )
        mock_wrapper_instance.load.assert_called_once()
        assert len(ds) == 15
        assert "text" in ds.column_names

    @patch("nightmarenet.data.ingest.DatasetWrapper")
    def test_from_huggingface_below_threshold_raises(self, mock_wrapper_class, ingestor):
        """from_huggingface should raise ValueError if dataset has < 10 samples."""
        mock_wrapper_instance = mock_wrapper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict({"text": [f"HF sample {i} content" for i in range(5)]})
        mock_wrapper_instance.train_data = mock_ds
        mock_wrapper_instance.load.return_value = mock_wrapper_instance

        with pytest.raises(ValueError, match="produced only 5 usable samples"):
            ingestor.from_huggingface("glue", subset="sst2")

    @patch("nightmarenet.data.ingest.DatasetWrapper")
    def test_from_huggingface_filters_empty_texts(self, mock_wrapper_class, ingestor):
        """from_huggingface should filter empty/whitespace-only texts."""
        mock_wrapper_instance = mock_wrapper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict(
            {
                "text": [
                    "Valid sample 1",
                    "   ",
                    "",
                    "Valid sample 2",
                    "\t\n",
                    "Valid sample 3",
                    "Valid sample 4",
                    "Valid sample 5",
                    "Valid sample 6",
                    "Valid sample 7",
                    "Valid sample 8",
                    "Valid sample 9",
                    "Valid sample 10",
                ]
            }
        )
        mock_wrapper_instance.train_data = mock_ds
        mock_wrapper_instance.load.return_value = mock_wrapper_instance

        ds = ingestor.from_huggingface("glue", subset="sst2")
        assert len(ds) == 10
        assert all(text.strip() for text in ds["text"])

    @patch("nightmarenet.data.ingest.DatasetWrapper")
    def test_from_huggingface_streaming_bypasses_finalise(self, mock_wrapper_class, ingestor):
        """streaming=True should bypass _finalise() and return wrapper.train_data directly."""
        mock_wrapper_instance = mock_wrapper_class.return_value
        from datasets import Dataset

        mock_ds = Dataset.from_dict({"text": [f"HF sample {i}" for i in range(3)]})
        mock_wrapper_instance.train_data = mock_ds
        mock_wrapper_instance.load.return_value = mock_wrapper_instance

        ds = ingestor.from_huggingface("glue", subset="sst2", streaming=True)
        assert ds == mock_ds
