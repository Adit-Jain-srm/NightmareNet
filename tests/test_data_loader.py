import unittest
from unittest import mock
import torch

from nightmarenet.data.loader import DatasetWrapper, VisionDatasetWrapper, load_from_config


class DummyHFDataset:
    def __init__(self, items, columns=None):
        self.items = items
        self.column_names = columns or ["text"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return [item[idx] for item in self.items]
        return self.items[idx]

    def train_test_split(self, test_size=0.1, seed=42):
        split_idx = int(len(self.items) * (1 - test_size))
        return {
            "train": DummyHFDataset(self.items[:split_idx], self.column_names),
            "test": DummyHFDataset(self.items[split_idx:], self.column_names),
        }

    def filter(self, fn):
        filtered = [item for item in self.items if fn(item)]
        return DummyHFDataset(filtered, self.column_names)

    def select(self, indices):
        selected = [self.items[i] for i in indices]
        return DummyHFDataset(selected, self.column_names)


class DummyStreamingHFDataset:
    def __init__(self, items, text_column="text"):
        self.items = items
        self.features = {text_column: "string"}
        self.text_column = text_column

    def filter(self, fn):
        filtered = [item for item in self.items if fn(item)]
        return DummyStreamingHFDataset(filtered, self.text_column)

    def take(self, n):
        return DummyStreamingHFDataset(self.items[:n], self.text_column)

    def __iter__(self):
        return iter(self.items)


class TestDataLoaderUnit(unittest.TestCase):
    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_load_from_config_valid_text(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset([{"text": "Sample 1"}, {"text": "Sample 2"}, {"text": "Sample 3"}]),
            "test": DummyHFDataset([{"text": "Test sample"}]),
        }
        mock_load_dataset.return_value = raw_mock

        config = {
            "dataset": {
                "name": "wikitext",
                "config": "wikitext-2-raw-v1",
                "text_column": "text",
            },
            "seed": 42,
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, DatasetWrapper)
        self.assertEqual(len(wrapper.get_texts("train")), 3)
        self.assertEqual(len(wrapper.get_texts("test")), 1)

    @mock.patch("torchvision.datasets.FakeData")
    def test_load_from_config_vision(self, mock_fakedata):
        mock_fakedata.return_value = [(torch.randn(3, 32, 32), 0)] * 10
        config = {
            "model": {"type": "image_classification"},
            "dataset": {"name": "cifar10", "max_samples": 5},
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, VisionDatasetWrapper)
        self.assertIsNotNone(wrapper.train_data)
        self.assertIsNotNone(wrapper.test_data)

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_max_samples_truncation(self, mock_load_dataset):
        items = [{"text": f"Text {i}"} for i in range(20)]
        raw_mock = {
            "train": DummyHFDataset(items[:15]),
            "test": DummyHFDataset(items[15:]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(
            dataset_name="dummy",
            text_column="text",
            max_samples=5,
        ).load()

        self.assertEqual(len(wrapper.train_data), 5)

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_streaming_mode(self, mock_load_dataset):
        items = [{"text": f"Stream {i}"} for i in range(10)]
        raw_mock = {
            "train": DummyStreamingHFDataset(items),
            "test": DummyStreamingHFDataset(items[:2]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(
            dataset_name="dummy",
            text_column="text",
            streaming=True,
            max_samples=4,
        ).load()

        self.assertIsNotNone(wrapper.train_data)
        with self.assertRaises(RuntimeError):
            wrapper.get_texts("train")

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_dataset_wrapper_unloaded_raises(self, mock_load_dataset):
        wrapper = DatasetWrapper(dataset_name="dummy")
        with self.assertRaises(RuntimeError):
            _ = wrapper.train_data
        with self.assertRaises(RuntimeError):
            _ = wrapper.test_data

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_invalid_max_samples_validation(self, mock_load_dataset):
        wrapper = DatasetWrapper(dataset_name="dummy", max_samples=-5)
        with self.assertRaises(ValueError):
            wrapper.load()

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_missing_text_column_validation(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset([{"wrong_col": "val"}]),
            "test": DummyHFDataset([{"wrong_col": "val"}]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(dataset_name="dummy", text_column="text")
        with self.assertRaises(ValueError):
            wrapper.load()


if __name__ == "__main__":
    unittest.main()
