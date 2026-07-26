import csv
import json
import os
from unittest import mock

import pytest

from nightmarenet.evaluation.failure_export import (
    export_failures_csv,
    export_failures_json,
    save_failure_report,
)


@pytest.fixture
def sample_failures():
    return [
        {
            "sample_index": 0,
            "original_input": "clean text 0",
            "distorted_input": "distorted text 0",
            "original_prediction": 1,
            "distorted_prediction": 1,
            "original_confidence": 0.9,
            "distorted_confidence": 0.8,
            "confidence_drop": 0.1,
            "distortion_type": "text_distortion",
            "distortion_strength": 0.5,
            "seed": 42,
        },
        {
            "sample_index": 1,
            "original_input": "clean text 1",
            "distorted_input": "distorted text 1",
            "original_prediction": 1,
            "distorted_prediction": 0,
            "original_confidence": 0.95,
            "distorted_confidence": 0.6,
            "confidence_drop": 0.35,
            "distortion_type": "text_distortion",
            "distortion_strength": 0.5,
            "seed": 42,
        },
        {
            "sample_index": 2,
            "original_input": "clean text 2",
            "distorted_input": "distorted text 2",
            "original_prediction": 0,
            "distorted_prediction": 0,
            "original_confidence": 0.8,
            "distorted_confidence": 0.4,
            "confidence_drop": 0.4,
            "distortion_type": "text_distortion",
            "distortion_strength": 0.5,
            "seed": 42,
        },
    ]


def test_export_disabled(tmp_path):
    # Testing that save_failure_report returns empty string if no failures matched
    filepath = save_failure_report([], str(tmp_path))
    assert filepath == ""


def test_json_export(tmp_path, sample_failures):
    filepath = tmp_path / "test_failures.json"
    export_failures_json(sample_failures, str(filepath))

    assert filepath.exists()
    with open(filepath) as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[1]["sample_index"] == 1


def test_csv_export(tmp_path, sample_failures):
    filepath = tmp_path / "test_failures.csv"
    export_failures_csv(sample_failures, str(filepath))

    assert filepath.exists()
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[1]["sample_index"] == "1"


def test_threshold_filtering(tmp_path, sample_failures):
    # sample 0: drop 0.1 (not exported if threshold is 0.2)
    # sample 2: drop 0.4, prediction not changed (exported due to drop > 0.2)
    filepath = save_failure_report(sample_failures, str(tmp_path), format="json", threshold=0.2)
    assert filepath != ""

    with open(filepath) as f:
        data = json.load(f)

    assert len(data) == 2
    indices = [d["sample_index"] for d in data]
    assert 1 in indices  # prediction changed
    assert 2 in indices  # confidence drop > threshold
    assert 0 not in indices  # neither


def test_prediction_change_detection(tmp_path, sample_failures):
    # sample 1 has prediction change. even if threshold is very high, it should be exported
    filepath = save_failure_report(sample_failures, str(tmp_path), format="json", threshold=0.99)
    assert filepath != ""

    with open(filepath) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["sample_index"] == 1


def test_expected_keys(tmp_path, sample_failures):
    filepath = save_failure_report(sample_failures, str(tmp_path), format="json", threshold=0.0)
    assert filepath != ""

    with open(filepath) as f:
        data = json.load(f)

    expected_keys = {
        "sample_index",
        "original_input",
        "distorted_input",
        "original_prediction",
        "distorted_prediction",
        "original_confidence",
        "distorted_confidence",
        "confidence_drop",
        "distortion_type",
        "distortion_strength",
        "seed",
    }
    assert set(data[0].keys()) == expected_keys


def test_empty_failures(tmp_path):
    filepath = tmp_path / "empty.csv"
    export_failures_csv([], str(filepath))
    # csv exporter returns early if empty
    assert not filepath.exists()


@mock.patch("nightmarenet.evaluation.failure_export.datetime")
def test_output_file_created(mock_datetime, tmp_path, sample_failures):
    mock_datetime.now.return_value.strftime.return_value = "20260101_120000"

    filepath = save_failure_report(sample_failures, str(tmp_path), format="csv", threshold=0.2)

    assert "20260101_120000_failures.csv" in filepath
    assert os.path.exists(filepath)
