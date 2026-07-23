"""EU AI Act Article 15 compliance report generation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    """Generate deterministic hash of configuration."""

    payload = json.dumps(config, sort_keys=True).encode()

    return hashlib.sha256(payload).hexdigest()


def _environment() -> dict[str, Any]:
    """Collect runtime environment information."""

    gpu = None

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "gpu": gpu,
    }


def _eu_ai_mapping() -> dict[str, Any]:
    """Map generated evidence to EU AI Act Article 15."""

    return {
        "article": "EU AI Act Article 15",
        "requirements": {
            "accuracy": "Measured through evaluation metrics.",
            "robustness": "Measured through robustness benchmark results.",
            "cybersecurity": (
                "Artifact hashes provide integrity verification "
                "for trained models and configuration."
            ),
        },
    }


def _nist_mapping() -> dict[str, Any]:
    """Map outputs to NIST AI RMF."""

    return {
        "Govern": [
            "Training lineage",
            "Configuration tracking",
        ],
        "Map": [
            "Dataset information",
            "Model metadata",
        ],
        "Measure": [
            "Evaluation metrics",
            "Robustness metrics",
        ],
        "Manage": [
            "Artifact integrity",
            "Reproducibility",
        ],
    }


def _build_report(
    config: dict[str, Any],
    comparison: dict[str, Any],
    model_path: str,
    tracker: Any = None,
) -> dict[str, Any]:
    """Build the compliance report dictionary."""

    config_hash = _config_hash(config)

    model_hash = None

    if model_path:
        model_path_obj = Path(model_path)

        if model_path_obj.is_file():
            model_hash = _sha256_file(str(model_path_obj))

        elif model_path_obj.is_dir():
            for filename in (
                "model.safetensors",
                "pytorch_model.bin",
                "model.pt",
                "checkpoint.pt",
            ):
                candidate = model_path_obj / filename
                if candidate.exists():
                    model_hash = _sha256_file(str(candidate))
                    break

    lineage = {}
    if tracker is not None:
        try:
            lineage = tracker.get_lineage()
        except AttributeError:
            lineage = {}

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})

    robustness_metrics = comparison.get("metrics", {}).get("robustness", {})

    trained = robustness_metrics.get("trained", {})
    deltas = robustness_metrics.get("deltas", {})

    robustness = {
        "clean_accuracy": trained.get("clean_accuracy"),
        "distorted_accuracy": trained.get("distorted_accuracy"),
        "auc_robustness": trained.get("auc_robustness"),
        "delta": deltas.get("auc_robustness"),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "model": {
            "name": model_config.get("name"),
            "type": model_config.get("type"),
        },
        "dataset": {
            "name": dataset_config.get("name"),
            "path": dataset_config.get("path"),
            "config": dataset_config,
        },
        "training_lineage": lineage,
        "artifact_integrity": {
            "config_sha256": config_hash,
            "model_sha256": model_hash,
        },
        "environment": _environment(),
        "robustness": robustness,
        "eu_ai_act": _eu_ai_mapping(),
        "nist_ai_rmf": _nist_mapping(),
    }

    return report


def _generate_markdown(report: dict[str, Any]) -> str:
    """Generate a human-readable markdown compliance report."""

    lines = [
        "# EU AI Act Article 15 Compliance Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Model",
        f"- Name: {report['model'].get('name')}",
        f"- Type: {report['model'].get('type')}",
        "",
        "## Artifact Integrity",
        f"- Config SHA-256: {report['artifact_integrity'].get('config_sha256')}",
        f"- Model SHA-256: {report['artifact_integrity'].get('model_sha256')}",
        "",
        "## Robustness",
        f"- Clean Accuracy: {report['robustness'].get('clean_accuracy')}",
        f"- Distorted Accuracy: {report['robustness'].get('distorted_accuracy')}",
        f"- AUC Robustness: {report['robustness'].get('auc_robustness')}",
        f"- Delta: {report['robustness'].get('delta')}",
        "",
        "## Runtime Environment",
        f"- Python: {report['environment'].get('python_version')}",
        f"- PyTorch: {report['environment'].get('pytorch_version')}",
        f"- GPU: {report['environment'].get('gpu')}",
        "",
        "## EU AI Act Mapping",
    ]

    for key, value in report["eu_ai_act"]["requirements"].items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## NIST AI RMF",
        ]
    )

    for section, items in report["nist_ai_rmf"].items():
        lines.append(f"### {section}")
        for item in items:
            lines.append(f"- {item}")

    return "\n".join(lines)


def generate_report(
    config: dict[str, Any],
    comparison: dict[str, Any],
    model_path: str,
    output_dir: str = "results",
    tracker: Any = None,
) -> dict[str, Any]:
    """Generate and save a compliance report.

    Creates both JSON and Markdown reports and returns the report dictionary.
    """

    report = _build_report(
        config=config,
        comparison=comparison,
        model_path=model_path,
        tracker=tracker,
    )

    os.makedirs(output_dir, exist_ok=True)

    run_id = tracker.run_id if tracker is not None else "latest"

    json_path = Path(output_dir) / f"{run_id}_compliance_report.json"
    md_path = Path(output_dir) / f"{run_id}_compliance_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_generate_markdown(report))

    return report


def generate_pdf(
    config: dict[str, Any],
    comparison: dict[str, Any],
    model_path: str,
    output_dir: str = "results",
    tracker: Any = None,
) -> str:
    """Generate a PDF compliance report with digital signature.

    Args:
        config: Training configuration dictionary.
        comparison: Comparison metrics dictionary.
        model_path: Path to the model file or directory.
        output_dir: Directory where the PDF will be saved.
        tracker: Optional tracker object for run ID.

    Returns:
        Path to the generated PDF file.

    Raises:
        ImportError: If PDF generation dependencies are not installed.
    """
    from nightmarenet.compliance.pdf_builder import generate_pdf as generate_pdf_impl

    report = _build_report(
        config=config,
        comparison=comparison,
        model_path=model_path,
        tracker=tracker,
    )

    os.makedirs(output_dir, exist_ok=True)

    run_id = tracker.run_id if tracker is not None else "latest"
    pdf_path = Path(output_dir) / f"{run_id}_compliance_report.pdf"

    return generate_pdf_impl(report, str(pdf_path), config=config)
