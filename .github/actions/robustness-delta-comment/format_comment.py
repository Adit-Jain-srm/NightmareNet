import json
import os


def format_percentage(val):
    return f"{val * 100:.1f}%"


def format_decimal(val):
    return f"{val:.3f}"


def format_delta(val, is_percentage=False):
    prefix = "+" if val > 0 else ""
    if is_percentage:
        return f"{prefix}{val * 100:.1f}%"
    return f"{prefix}{val:.3f}"


def main():
    try:
        with open("delta_results.json") as f:
            data = json.load(f)
    except Exception:
        # Fallback if compute_delta didn't run or file is missing
        model_exists = os.environ.get("MODEL_EXISTS", "true").lower() == "true"
        reason = (
            "Evaluation timed out or failed"
            if model_exists
            else "Evaluation skipped (no model found)"
        )
        data = {"skipped": True, "reason": reason}

    skipped = data.get("skipped", False)
    if skipped:
        reason = data.get("reason", "Evaluation skipped (no model found)")
        lines = [
            "<!-- robustness-delta-comment -->",
            "## Robustness Regression Report",
            "",
            f"_{reason}_",
        ]
        with open("comment_body.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    threshold = data.get("threshold", -0.05)
    results = data.get("results", [])
    exceeds_threshold = data.get("exceeds_threshold", False)

    lines = [
        "<!-- robustness-delta-comment -->",
        "## Robustness Regression Report",
        "",
        "| Metric | main | This PR | Delta | Status |",
        "|--------|------|---------|-------|--------|",
    ]

    for item in results:
        metric = item["metric"]
        # Format as percentage if name contains accuracy or textfooler, else decimal
        # We assume overall robustness is decimal, others can be percentage based on name,
        # but let's check values (if <= 1, maybe % is good). Let's use % for clean_accuracy,
        # and decimal for robustness_score as per issue example.
        is_percent = "accuracy" in metric.lower() or any(
            x in metric.lower() for x in ["fooler", "attack", "bugg"]
        )

        main_str = format_percentage(item["main"]) if is_percent else format_decimal(item["main"])
        pr_str = format_percentage(item["pr"]) if is_percent else format_decimal(item["pr"])
        delta_str = format_delta(item["delta"], is_percent)

        # Display name tweaks
        display_metric = metric.replace("_", " ").title()
        if "Textfooler" in display_metric:
            display_metric = display_metric.replace("Textfooler", "TextFooler")
        if "Bertattack" in display_metric:
            display_metric = display_metric.replace("Bertattack", "BertAttack")
        if "Robustness Score" in display_metric:
            display_metric = "Overall Robustness"

        status = item["status"]
        if status == "Pass":
            status_str = "✅ Pass"
        elif status == "Warning":
            status_str = "⚠️ Warning"
        else:
            status_str = "❌ Fail"

        lines.append(f"| {display_metric} | {main_str} | {pr_str} | {delta_str} | {status_str} |")

    lines.append("")
    if exceeds_threshold:
        thresh_str = format_percentage(threshold) if threshold > -1 and threshold < 1 else threshold
        lines.append(
            f"**Verdict: FAIL** (one or more metrics regressed beyond threshold of {thresh_str})"
        )
    else:
        thresh_str = format_percentage(threshold) if threshold > -1 and threshold < 1 else threshold
        lines.append(f"**Verdict: PASS** (no metric regressed beyond threshold of {thresh_str})")

    with open("comment_body.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
