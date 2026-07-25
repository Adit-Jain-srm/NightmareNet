"""Module for exporting robustness evaluation failures."""

import csv
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def export_failures_json(failures: list[dict], filepath: str) -> None:
    """Export failures to a JSON file.

    Args:
        failures: List of failure dictionaries.
        filepath: Path to the output JSON file.
    """
    try:
        with open(filepath, "w") as f:
            json.dump(failures, f, indent=2, default=str)
        logger.info("Exported %d failures to %s", len(failures), filepath)
    except Exception as e:
        logger.error("Failed to export JSON failures to %s: %s", filepath, e)


def export_failures_csv(failures: list[dict], filepath: str) -> None:
    """Export failures to a CSV file.

    Args:
        failures: List of failure dictionaries.
        filepath: Path to the output CSV file.
    """
    if not failures:
        return

    try:
        keys = list(failures[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(failures)
        logger.info("Exported %d failures to %s", len(failures), filepath)
    except Exception as e:
        logger.error("Failed to export CSV failures to %s: %s", filepath, e)


def save_failure_report(
    failures: list[dict], output_dir: str, format: str = "json", threshold: float = 0.20
) -> str:
    """Filter and save failure report.

    Args:
        failures: List of all per-sample data collected during robustness evaluation.
        output_dir: Directory to save the failure report.
        format: Export format ('json' or 'csv').
        threshold: Confidence drop threshold for failure definition.

    Returns:
        The path to the saved report, or empty string if no failures.
    """
    os.makedirs(output_dir, exist_ok=True)

    failed_samples = []
    for sample in failures:
        prediction_changed = sample.get("original_prediction") != sample.get("distorted_prediction")
        confidence_drop = sample.get("confidence_drop", 0.0)

        if prediction_changed or confidence_drop > threshold:
            failed_samples.append(sample)

    if not failed_samples:
        logger.info("No failures detected. Skipping export.")
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{timestamp}_failures.{format}")

    if format.lower() == "csv":
        export_failures_csv(failed_samples, filepath)
    else:
        export_failures_json(failed_samples, filepath)

    return filepath
