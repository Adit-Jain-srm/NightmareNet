"""Output formatters for ensemble benchmarking and certification results (issue #162)."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Any, Optional


def to_json(results: dict[str, Any], output_path: str) -> None:
    """Export results to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def to_csv(models_summary: list[dict[str, Any]], output_path: str) -> None:
    """Export summary metrics to CSV."""
    if not models_summary:
        return
    keys = list(models_summary[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(models_summary)


def to_latex_table(
    models_summary: list[dict[str, Any]],
    output_path: str,
    certification: Optional[dict[str, Any]] = None,
) -> None:
    """Generate a LaTeX table for the models summary.

    Args:
        models_summary: Per-model rows (see to_csv).
        output_path: Where to write the .tex file.
        certification: Optional aggregate certification stats (the dict shape
            returned by certification.certify_dataset / Evaluator._run_certification,
            i.e. with certified_radius_mean / certified_radius_median /
            certification_abstain_rate / samples_certified keys). When given, an
            extra "Certified Robustness (Randomized Smoothing)" block is appended
            below the main table -- issue #162's "add certification row to the
            results table". Kept as a separate, clearly-labeled block rather than
            columns on the per-model table since certification is a per-run
            aggregate (formal, distribution-free guarantee), not a per-model
            empirical score -- conflating the two into one row would misrepresent
            what each number claims.
    """
    if not models_summary and certification is None:
        return

    lines: list[str] = []

    if models_summary:
        lines.extend(
            [
                "\\begin{table}[h]",
                "\\centering",
                "\\begin{tabular}{lccc}",
                "\\hline",
                "\\textbf{Model} & \\textbf{Robustness Score} & ",
                "\\textbf{Latency (s)} & \\textbf{Parameters} \\\\",
                "\\hline",
            ]
        )

        for m in models_summary:
            model_name = str(m.get("model", "Unknown")).replace("_", "\\_")
            rob = m.get("robustness", 0.0)
            lat = m.get("latency", 0.0)
            params = m.get("params", 0)

            # Format params nicely, e.g., 110M
            if params >= 1_000_000:
                params_str = f"{params / 1_000_000:.1f}M"
            elif params >= 1_000:
                params_str = f"{params / 1_000:.1f}K"
            else:
                params_str = str(params)

            lines.append(f"{model_name} & {rob:.4f} & {lat:.4f} & {params_str} \\\\")

        lines.extend(
            [
                "\\hline",
                "\\end{tabular}",
                "\\caption{Ensemble Robustness Benchmarking Results}",
                "\\label{tab:ensemble_results}",
                "\\end{table}",
            ]
        )

    if certification is not None:
        if lines:
            lines.append("")
        radius_mean = certification.get("certified_radius_mean", 0.0)
        radius_median = certification.get("certified_radius_median", 0.0)
        abstain_rate = certification.get("certification_abstain_rate", 0.0)
        n_samples = certification.get("samples_certified", certification.get("n_samples", 0))
        certified_accuracy = certification.get("certified_accuracy")
        accuracy_str = (
            f"{certified_accuracy:.4f}" if isinstance(certified_accuracy, (int, float)) else "N/A"
        )

        lines.extend(
            [
                "\\begin{table}[h]",
                "\\centering",
                "\\begin{tabular}{lc}",
                "\\hline",
                "\\textbf{Certified Robustness Metric} & \\textbf{Value} \\\\",
                "\\hline",
                f"Mean certified radius (L2, embedding space) & {radius_mean:.4f} \\\\",
                f"Median certified radius (L2, embedding space) & {radius_median:.4f} \\\\",
                f"Abstention rate & {abstain_rate * 100:.1f}\\% \\\\",
                f"Certified accuracy & {accuracy_str} \\\\",
                f"Samples certified & {n_samples} \\\\",
                "\\hline",
                "\\end{tabular}",
                "\\caption{Certified Robustness via Randomized Smoothing "
                "(Cohen et al., 2019). Radii are formal, distribution-free "
                "lower bounds in embedding space -- not directly comparable to "
                "the empirical robustness score above.}",
                "\\label{tab:certified_robustness}",
                "\\end{table}",
            ]
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _certification_result_to_row(index: int, result: Any) -> dict[str, Any]:
    """Flattens one certification result (dataclass or dict) into a CSV-ready row.

    Accepts either a certification.CertificationResult dataclass instance (the
    shape certify_dataset's "results" list actually contains) or an already-plain
    dict (e.g. if the caller pre-serialized it) -- so this can be handed either
    the raw certify_dataset() output or a JSON round-tripped copy of it.
    """
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        row = dataclasses.asdict(result)
    elif isinstance(result, dict):
        row = dict(result)
    else:
        raise TypeError(
            f"Expected a CertificationResult dataclass or dict, got {type(result).__name__}"
        )
    return {"sample_index": index, **row}


def certification_to_json_dict(cert_result: dict[str, Any]) -> dict[str, Any]:
    """Builds the JSON-serializable "certification" block (issue #162).

    Takes the raw dict returned by certification.certify_dataset (aggregates plus a
    "results" list of CertificationResult dataclass instances) and converts it into
    a plain-dict shape with a per-sample radii array alongside the aggregates, ready
    to merge into any results dict passed to to_json -- e.g.
    `results["certification"] = certification_to_json_dict(cert_result)`.

    Kept separate from to_json (which stays a generic passthrough dumper) so
    certify_dataset's dataclass "results" entries get serialized deliberately
    rather than relying on to_json's `default=str` fallback, which would collapse
    each CertificationResult into an opaque repr string instead of a structured,
    queryable per-sample record.
    """
    samples = [
        _certification_result_to_row(i, r) for i, r in enumerate(cert_result.get("results", []))
    ]
    return {
        "metric": "certification",
        "n_samples": cert_result.get("n_samples", len(samples)),
        "certified_radius_mean": cert_result.get("certified_radius_mean", 0.0),
        "certified_radius_median": cert_result.get("certified_radius_median", 0.0),
        "certification_abstain_rate": cert_result.get("certification_abstain_rate", 0.0),
        "certified_accuracy": cert_result.get("certified_accuracy"),
        "samples": samples,
    }


def to_csv_certification(cert_result: dict[str, Any], output_path: str) -> None:
    """Exports per-sample certification results to CSV (issue #162).

    One row per certified sample, with `certified_radius` and `abstained` among the
    columns (per the issue's acceptance criteria) plus the rest of each
    CertificationResult's fields (prediction, p_a_lower, n_samples_used, label,
    correct, sample_index) for full traceability back to the source sample.

    Args:
        cert_result: Raw certify_dataset() output (must contain a "results" list).
        output_path: Where to write the CSV.
    """
    results = cert_result.get("results", [])
    if not results:
        return

    rows = [_certification_result_to_row(i, r) for i, r in enumerate(results)]
    # Stable, predictable column order regardless of dataclass field order, with the
    # two columns the issue calls out explicitly (certified_radius, abstained) first.
    priority_cols = ["sample_index", "certified_radius", "abstained"]
    remaining_cols = [k for k in rows[0] if k not in priority_cols]
    fieldnames = priority_cols + remaining_cols

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_all(
    results: dict[str, Any],
    formats: list[str],
    output_dir: str,
    prefix: str = "ensemble",
    certification: Optional[dict[str, Any]] = None,
) -> None:
    """Export to all requested formats.

    Args:
        results: Ensemble-style results dict (reads "models_summary" for
            table generation, as before).
        formats: Any of "json", "csv", "latex".
        output_dir: Directory to write outputs into.
        prefix: Output filename prefix.
        certification: Optional raw certify_dataset()-shaped dict (issue #162).
            When given: merged into the JSON export under a "certification" key
            (per-sample radii array + aggregates), written to its own
            `{prefix}_certification.csv` (one row per sample), and appended as an
            extra block in the LaTeX table output. Omitted entirely (no behavior
            change) when None, so existing ensemble-only callers are unaffected.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "json" in formats:
        json_results = results
        if certification is not None:
            json_results = {**results, "certification": certification_to_json_dict(certification)}
        to_json(json_results, os.path.join(output_dir, f"{prefix}_results.json"))

    models_summary = results.get("models_summary", [])
    if "csv" in formats:
        to_csv(models_summary, os.path.join(output_dir, f"{prefix}_summary.csv"))
        if certification is not None:
            to_csv_certification(
                certification, os.path.join(output_dir, f"{prefix}_certification.csv")
            )

    if "latex" in formats:
        to_latex_table(
            models_summary,
            os.path.join(output_dir, f"{prefix}_table.tex"),
            certification=certification,
        )
