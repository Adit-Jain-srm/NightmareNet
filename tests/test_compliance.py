import json
from pathlib import Path

from nightmarenet.compliance.report import generate_report


def test_generate_compliance_report(tmp_path):
    config = {
        "model": {
            "name": "test-model",
            "type": "transformer",
        },
        "dataset": {
            "name": "dummy",
            "path": "dummy/path",
        },
    }

    comparison = {
        "robustness_score": 0.91,
        "robustness_delta": 0.07,
        "metrics": {
            "robustness": {
                "trained": {
                    "auc_robustness": 0.91,
                },
                "deltas": {
                    "auc_robustness": 0.07,
                },
            }
        },
    }

    report = generate_report(
        config=config,
        comparison=comparison,
        model_path="",
        output_dir=str(tmp_path),
    )

    assert report["model"]["name"] == "test-model"
    assert report["robustness"]["delta"] == 0.07

    files = list(Path(tmp_path).glob("*compliance_report.json"))
    assert files

    with open(files[0], encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["model"]["name"] == "test-model"


def test_config_hash_is_deterministic():
    from nightmarenet.compliance.report import _config_hash

    config = {
        "model": {"name": "demo"},
        "dataset": {"name": "dummy"},
    }

    first = _config_hash(config)
    second = _config_hash(config)

    assert first == second


def test_config_defaults_to_no_compliance_report():
    config = {
        "tracking": {
            "compliance_report": False,
        }
    }

    assert config["tracking"]["compliance_report"] is False
