"""Tests for nightmarenet.evaluation.format_results (issue #162: certification export)."""

from __future__ import annotations

import csv
import json
import os

from nightmarenet.evaluation.certification import CertificationResult
from nightmarenet.evaluation.format_results import (
    certification_to_json_dict,
    format_all,
    to_csv_certification,
    to_latex_table,
)


def _sample_cert_result() -> dict:
    results = [
        CertificationResult(
            prediction=1,
            certified_radius=0.15,
            p_a_lower=0.91,
            n_samples_used=1100,
            abstained=False,
            label=1,
            correct=True,
        ),
        CertificationResult(
            prediction=0,
            certified_radius=0.0,
            p_a_lower=0.30,
            n_samples_used=1100,
            abstained=True,
            label=1,
            correct=False,
        ),
    ]
    return {
        "metric": "certification",
        "n_samples": 2,
        "certified_radius_mean": 0.075,
        "certified_radius_median": 0.075,
        "certification_abstain_rate": 0.5,
        "certified_accuracy": 0.5,
        "results": results,
    }


class TestCertificationToJsonDict:
    def test_includes_aggregates_and_per_sample_array(self):
        d = certification_to_json_dict(_sample_cert_result())

        assert d["metric"] == "certification"
        assert d["n_samples"] == 2
        assert d["certified_radius_mean"] == 0.075
        assert len(d["samples"]) == 2
        assert d["samples"][0]["certified_radius"] == 0.15
        assert d["samples"][0]["abstained"] is False
        assert d["samples"][1]["abstained"] is True

    def test_is_json_serializable(self):
        """The whole point of not relying on to_json's default=str fallback --
        dataclass instances must convert to plain dicts, not opaque repr strings."""
        d = certification_to_json_dict(_sample_cert_result())
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        assert reloaded["samples"][0]["prediction"] == 1

    def test_empty_results_list(self):
        empty = {
            "metric": "certification",
            "n_samples": 0,
            "certified_radius_mean": 0.0,
            "certified_radius_median": 0.0,
            "certification_abstain_rate": 0.0,
            "certified_accuracy": None,
            "results": [],
        }
        d = certification_to_json_dict(empty)
        assert d["samples"] == []
        assert d["certified_accuracy"] is None


class TestToCsvCertification:
    def test_writes_one_row_per_sample_with_required_columns(self, tmp_path):
        out_path = os.path.join(tmp_path, "cert.csv")
        to_csv_certification(_sample_cert_result(), out_path)

        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        # Issue #162 explicitly calls out these two columns.
        assert "certified_radius" in rows[0]
        assert "abstained" in rows[0]
        assert rows[0]["certified_radius"] == "0.15"
        assert rows[0]["abstained"] == "False"
        assert rows[1]["abstained"] == "True"

    def test_priority_columns_come_first(self, tmp_path):
        out_path = os.path.join(tmp_path, "cert.csv")
        to_csv_certification(_sample_cert_result(), out_path)

        with open(out_path, newline="") as f:
            header = next(csv.reader(f))

        assert header[:3] == ["sample_index", "certified_radius", "abstained"]

    def test_no_file_written_when_no_results(self, tmp_path):
        out_path = os.path.join(tmp_path, "cert.csv")
        to_csv_certification({"results": []}, out_path)
        assert not os.path.exists(out_path)


class TestToLatexTableCertification:
    def test_certification_block_appended_when_provided(self, tmp_path):
        out_path = os.path.join(tmp_path, "table.tex")
        to_latex_table([], out_path, certification=_sample_cert_result())

        content = open(out_path).read()
        assert "Certified Robustness" in content
        assert "0.0750" in content  # mean radius
        assert "50.0\\%" in content  # abstention rate

    def test_backward_compatible_without_certification(self, tmp_path):
        """Existing callers (e.g. ensemble_benchmark) that never pass `certification`
        must see identical output to before."""
        out_path = os.path.join(tmp_path, "table.tex")
        models_summary = [
            {"model": "distilbert", "robustness": 0.66, "latency": 0.1, "params": 66_000_000}
        ]
        to_latex_table(models_summary, out_path)

        content = open(out_path).read()
        assert "Certified Robustness" not in content
        assert "distilbert" in content

    def test_certification_only_no_models_summary(self, tmp_path):
        out_path = os.path.join(tmp_path, "table.tex")
        to_latex_table([], out_path, certification=_sample_cert_result())
        assert os.path.exists(out_path)

    def test_nothing_written_when_both_empty(self, tmp_path):
        out_path = os.path.join(tmp_path, "table.tex")
        to_latex_table([], out_path)
        assert not os.path.exists(out_path)

    def test_none_certified_accuracy_renders_as_na(self, tmp_path):
        cert = _sample_cert_result()
        cert["certified_accuracy"] = None
        out_path = os.path.join(tmp_path, "table.tex")
        to_latex_table([], out_path, certification=cert)

        content = open(out_path).read()
        assert "N/A" in content


class TestFormatAllCertification:
    def test_json_export_includes_certification_key(self, tmp_path):
        out_dir = os.path.join(tmp_path, "out")
        format_all(
            {"models_summary": []},
            formats=["json"],
            output_dir=out_dir,
            prefix="test",
            certification=_sample_cert_result(),
        )
        data = json.load(open(os.path.join(out_dir, "test_results.json")))
        assert "certification" in data
        assert len(data["certification"]["samples"]) == 2

    def test_csv_export_writes_certification_file(self, tmp_path):
        out_dir = os.path.join(tmp_path, "out")
        format_all(
            {"models_summary": []},
            formats=["csv"],
            output_dir=out_dir,
            prefix="test",
            certification=_sample_cert_result(),
        )
        assert os.path.exists(os.path.join(out_dir, "test_certification.csv"))

    def test_latex_export_includes_certification_block(self, tmp_path):
        out_dir = os.path.join(tmp_path, "out")
        format_all(
            {"models_summary": []},
            formats=["latex"],
            output_dir=out_dir,
            prefix="test",
            certification=_sample_cert_result(),
        )
        content = open(os.path.join(out_dir, "test_table.tex")).read()
        assert "Certified Robustness" in content

    def test_ensemble_only_unaffected_when_certification_omitted(self, tmp_path):
        """Regression guard: default `certification=None` must not change any
        existing ensemble-only behavior or output filenames."""
        out_dir = os.path.join(tmp_path, "out")
        models_summary = [{"model": "m1", "robustness": 0.5, "latency": 0.2, "params": 1000}]
        format_all(
            {"models_summary": models_summary},
            formats=["json", "csv", "latex"],
            output_dir=out_dir,
            prefix="ensemble",
        )
        files = set(os.listdir(out_dir))
        assert files == {"ensemble_results.json", "ensemble_summary.csv", "ensemble_table.tex"}
