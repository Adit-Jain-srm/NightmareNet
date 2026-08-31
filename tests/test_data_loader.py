"""Unit tests for nightmarenet.data.loader."""

import unittest
from unittest import mock

import torch
from datasets import IterableDataset

from nightmarenet.data.loader import (
    DatasetWrapper,
    VisionDatasetWrapper,
    VisionItemWrapper,
    load_from_config,
)


class DummyHFDataset:
    def __init__(self, items, columns=None):
        self.items = list(items)
        self.column_names = columns or (list(items[0].keys()) if items else ["text"])

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


class DummyStreamingHFDataset(IterableDataset):
    def __init__(self, items, text_column="text", features=None):
        self.items = list(items)
        self.text_column = text_column
        self._features = features if features is not None else {text_column: "string"}

    @property
    def features(self):
        return self._features

    def filter(self, fn):
        filtered = [item for item in self.items if fn(item)]
        return DummyStreamingHFDataset(filtered, self.text_column, self.features)

    def take(self, n):
        return DummyStreamingHFDataset(self.items[:n], self.text_column, self.features)

    def __iter__(self):
        return iter(self.items)


class TestDataLoaderUnit(unittest.TestCase):
    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_load_from_config_valid_text(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset(
                [{"text": "Sample 1"}, {"text": "Sample 2"}, {"text": "Sample 3"}]
            ),
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

    @mock.patch("torchvision.datasets.CIFAR10")
    def test_load_from_config_vision_cifar10(self, mock_cifar10):
        mock_data = [(torch.randn(3, 32, 32), 0)] * 10
        mock_cifar10.return_value = mock_data

        config = {
            "model": {"type": "image_classification"},
            "dataset": {"name": "cifar10", "max_samples": 5},
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, VisionDatasetWrapper)
        self.assertIsNotNone(wrapper.train_data)
        self.assertIsNotNone(wrapper.test_data)
        self.assertEqual(len(wrapper.train_data), 5)
        self.assertEqual(len(wrapper.test_data), 1)

    @mock.patch("torchvision.datasets.FakeData")
    @mock.patch(
        "torchvision.datasets.CIFAR10",
        side_effect=RuntimeError("CIFAR-10 download failed"),
    )
    def test_load_from_config_vision_cifar10_fallback(self, mock_cifar10, mock_fakedata):
        mock_fakedata.return_value = [(torch.randn(3, 32, 32), 1)] * 10
        config = {
            "model": {"type": "image_classification"},
            "dataset": {"name": "cifar10", "max_samples": 4},
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, VisionDatasetWrapper)
        self.assertEqual(len(wrapper.train_data), 4)

    @mock.patch("torchvision.datasets.ImageFolder")
    @mock.patch("os.path.isdir", return_value=True)
    def test_load_from_config_vision_imagenet_existing_path(self, mock_isdir, mock_imagefolder):
        mock_imagefolder.return_value = [(torch.randn(3, 224, 224), 2)] * 15
        config = {
            "model": {"type": "image_classification"},
            "dataset": {"name": "imagenet", "path": "/fake/imagenet", "max_samples": 5},
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, VisionDatasetWrapper)
        self.assertEqual(len(wrapper.train_data), 5)

    @mock.patch("torchvision.datasets.FakeData")
    def test_load_from_config_vision_unknown_dataset(self, mock_fakedata):
        mock_fakedata.return_value = [(torch.randn(3, 32, 32), 3)] * 10
        config = {
            "model": {"type": "image_classification"},
            "dataset": {"name": "unknown_dataset", "max_samples": 3},
        }

        wrapper = load_from_config(config)
        self.assertIsInstance(wrapper, VisionDatasetWrapper)
        self.assertEqual(len(wrapper.train_data), 3)

    def test_vision_item_wrapper_tensor_conversion(self):
        mock_raw_data = [(torch.randn(3, 32, 32), 0)]
        wrapper = VisionItemWrapper(mock_raw_data)
        self.assertEqual(len(wrapper), 1)
        item = wrapper[0]
        self.assertIn("pixel_values", item)
        self.assertIn("labels", item)
        self.assertTrue(isinstance(item["pixel_values"], torch.Tensor))
        self.assertEqual(item["labels"], 0)

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
        self.assertEqual(len(wrapper.test_data), 1)

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
    def test_streaming_mode_missing_text_column(self, mock_load_dataset):
        items = [{"wrong_col": f"Stream {i}"} for i in range(5)]
        raw_mock = {
            "train": DummyStreamingHFDataset(
                items, text_column="wrong_col", features={"other_col": "string"}
            ),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(
            dataset_name="dummy",
            text_column="text",
            streaming=True,
        )
        with self.assertRaises(ValueError):
            wrapper.load()

    def test_dataset_wrapper_unloaded_raises(self):
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

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_filter_empty_and_whitespace_texts(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset(
                [{"text": "Valid"}, {"text": ""}, {"text": "   "}, {"text": "Also valid"}]
            ),
            "test": DummyHFDataset([{"text": "Valid test"}, {"text": ""}]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(dataset_name="dummy", text_column="text").load()
        self.assertEqual(len(wrapper.train_data), 2)
        self.assertEqual(len(wrapper.test_data), 1)

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_glue_dataset_fallback(self, mock_load_dataset):
        mock_load_dataset.side_effect = [
            RuntimeError("nyu-mll/glue failed"),
            {
                "train": DummyHFDataset([{"text": "Sample 1"}]),
                "validation": DummyHFDataset([{"text": "Val 1"}]),
            },
        ]
        wrapper = DatasetWrapper(dataset_name="glue", subset="sst2", text_column="text").load()
        self.assertEqual(len(wrapper.train_data), 1)
        self.assertEqual(len(wrapper.test_data), 1)

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_load_dataset_failure_raises_runtime_error(self, mock_load_dataset):
        mock_load_dataset.side_effect = RuntimeError("Network error")
        wrapper = DatasetWrapper(dataset_name="nonexistent")
        with self.assertRaises(RuntimeError):
            wrapper.load()

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_train_test_split_created_when_no_test_split(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset([{"text": f"Sample {i}"} for i in range(10)]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(dataset_name="dummy", text_column="text").load()
        self.assertEqual(len(wrapper.train_data), 9)
        self.assertEqual(len(wrapper.test_data), 1)

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_deterministic_shuffling_and_splitting(self, mock_load_dataset):
        def make_raw():
            return {
                "train": DummyHFDataset([{"text": f"Sample {i}"} for i in range(10)]),
            }

        mock_load_dataset.side_effect = [make_raw(), make_raw()]
        wrapper1 = DatasetWrapper(dataset_name="dummy", text_column="text", seed=123).load()
        wrapper2 = DatasetWrapper(dataset_name="dummy", text_column="text", seed=123).load()

        self.assertEqual(wrapper1.get_texts("train"), wrapper2.get_texts("train"))
        self.assertEqual(wrapper1.get_texts("test"), wrapper2.get_texts("test"))

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_empty_dataset_handling(self, mock_load_dataset):
        raw_mock = {
            "train": DummyHFDataset([{"text": ""}, {"text": "   "}]),
            "test": DummyHFDataset([{"text": "  \n\t "}]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(dataset_name="dummy", text_column="text").load()
        self.assertEqual(len(wrapper.train_data), 0)
        self.assertEqual(len(wrapper.test_data), 0)
        self.assertEqual(wrapper.get_texts("train"), [])
        self.assertEqual(wrapper.get_texts("test"), [])

    @mock.patch("nightmarenet.data.loader.load_dataset")
    def test_dataset_wrapper_iteration(self, mock_load_dataset):
        samples = [{"text": f"Line {i}"} for i in range(5)]
        raw_mock = {
            "train": DummyHFDataset(samples[:4]),
            "test": DummyHFDataset(samples[4:]),
        }
        mock_load_dataset.return_value = raw_mock

        wrapper = DatasetWrapper(dataset_name="dummy", text_column="text").load()
        iterated = [item["text"] for item in wrapper.train_data]
        self.assertEqual(iterated, ["Line 0", "Line 1", "Line 2", "Line 3"])
        self.assertEqual(wrapper.train_data[0]["text"], "Line 0")


if __name__ == "__main__":
    unittest.main()
