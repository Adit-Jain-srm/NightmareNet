import unittest

from fastapi.testclient import TestClient

from nightmarenet.api.app import app
from nightmarenet.utils.validation import (
    validate_config_keys,
    validate_dataloader,
    validate_dataset_columns,
    validate_non_empty_dataset,
    validate_positive_float,
    validate_positive_int,
    validate_ratio,
    validate_strength,
    validate_text,
)

client = TestClient(app)


def test_webhook_settings_validation_fails_on_bad_data():
    bad_payload = {"webhooks": "this_is_not_a_valid_list"}
    response = client.post("/api/v1/settings/webhooks", json=bad_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


class TestValidationUtils(unittest.TestCase):
    def test_validate_strength_valid(self):
        self.assertEqual(validate_strength(0.0), 0.0)
        self.assertEqual(validate_strength(0.5), 0.5)
        self.assertEqual(validate_strength(1.0), 1.0)
        self.assertEqual(validate_strength(1), 1.0)

    def test_validate_strength_invalid(self):
        with self.assertRaises(ValueError):
            validate_strength(-0.1)
        with self.assertRaises(ValueError):
            validate_strength(1.1)
        with self.assertRaises(TypeError):
            validate_strength("0.5")  # type: ignore

    def test_validate_positive_int_valid(self):
        self.assertEqual(validate_positive_int(1), 1)
        self.assertEqual(validate_positive_int(100), 100)
        self.assertEqual(validate_positive_int(0, allow_zero=True), 0)

    def test_validate_positive_int_invalid(self):
        with self.assertRaises(ValueError):
            validate_positive_int(0)
        with self.assertRaises(ValueError):
            validate_positive_int(-5)
        with self.assertRaises(TypeError):
            validate_positive_int(1.5)  # type: ignore
        with self.assertRaises(TypeError):
            validate_positive_int(True)  # type: ignore

    def test_validate_positive_float(self):
        self.assertEqual(validate_positive_float(0.1), 0.1)
        self.assertEqual(validate_positive_float(0, allow_zero=True), 0.0)
        with self.assertRaises(ValueError):
            validate_positive_float(0.0)
        with self.assertRaises(TypeError):
            validate_positive_float("abc")  # type: ignore

    def test_validate_ratio(self):
        self.assertEqual(validate_ratio(0.0), 0.0)
        self.assertEqual(validate_ratio(0.999), 0.999)
        with self.assertRaises(ValueError):
            validate_ratio(1.0)
        with self.assertRaises(ValueError):
            validate_ratio(-0.1)

    def test_validate_text(self):
        self.assertEqual(validate_text("hello"), "hello")
        self.assertEqual(validate_text("", allow_empty=True), "")
        with self.assertRaises(ValueError):
            validate_text("", allow_empty=False)
        with self.assertRaises(TypeError):
            validate_text(123)  # type: ignore

    def test_validate_dataset_columns(self):
        class DummyDS:
            column_names = ["text", "label"]

        validate_dataset_columns(DummyDS(), ["text"])
        with self.assertRaises(ValueError):
            validate_dataset_columns(DummyDS(), ["text", "missing_col"])
        with self.assertRaises(AttributeError):
            validate_dataset_columns(object(), ["text"])

    def test_validate_non_empty_dataset(self):
        validate_non_empty_dataset([1, 2, 3])
        with self.assertRaises(ValueError):
            validate_non_empty_dataset([])

    def test_validate_config_keys(self):
        config = {"a": 1, "b": 2}
        validate_config_keys(config, ["a"])
        with self.assertRaises(ValueError):
            validate_config_keys(config, ["a", "c"])
        with self.assertRaises(TypeError):
            validate_config_keys([], ["a"])  # type: ignore

    def test_validate_dataloader(self):
        validate_dataloader(object())
        with self.assertRaises(ValueError):
            validate_dataloader(None)


if __name__ == "__main__":
    unittest.main()
