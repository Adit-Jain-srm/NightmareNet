import json
import os
import sys


def main():
    pr_file = os.environ.get("PR_RESULTS")
    main_file = os.environ.get("MAIN_RESULTS")
    threshold = float(os.environ.get("THRESHOLD", "-0.05"))
    metrics_str = os.environ.get("METRICS", "clean_accuracy,robustness_score,textfooler_0.5")
    metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]

    # Check if results exist
    pr_exists = pr_file and os.path.exists(pr_file)
    main_exists = main_file and os.path.exists(main_file)
    model_exists = os.environ.get("MODEL_EXISTS", "true").lower() == "true"

    if not pr_exists or not main_exists:
        reason = "Evaluation skipped (no model found)"
        if model_exists:
            reason = "Evaluation timed out or failed"

        with open("delta_results.json", "w") as f:
            json.dump(
                {
                    "skipped": True,
                    "reason": reason,
                    "threshold": threshold,
                    "results": [],
                    "exceeds_threshold": False,
                },
                f,
            )

        # Export variables to GITHUB_ENV
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as f:
                f.write("EXCEEDS_THRESHOLD=false\n")
                f.write("HAS_REGRESSION=false\n")
                f.write("HAS_SKIPPED=true\n")
        sys.exit(0)

    try:
        with open(pr_file) as f:
            pr_data = json.load(f)
        with open(main_file) as f:
            main_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"::error::Failed to load result files: {e}\n")
        sys.exit(1)

    results = []
    exceeds_threshold = False
    has_regression = False

    for metric in metrics:
        pr_val = pr_data.get(metric)
        main_val = main_data.get(metric)

        if pr_val is None or main_val is None:
            continue

        pr_val = float(pr_val)
        main_val = float(main_val)
        delta = pr_val - main_val

        if delta < 0:
            has_regression = True
            if delta < threshold:
                exceeds_threshold = True

        results.append(
            {
                "metric": metric,
                "main": main_val,
                "pr": pr_val,
                "delta": delta,
                "status": "Fail" if delta < threshold else ("Warning" if delta < 0 else "Pass"),
            }
        )

    with open("delta_results.json", "w") as f:
        json.dump(
            {"threshold": threshold, "results": results, "exceeds_threshold": exceeds_threshold},
            f,
        )

    # Export variables for subsequent steps
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"EXCEEDS_THRESHOLD={'true' if exceeds_threshold else 'false'}\n")
        f.write(f"HAS_REGRESSION={'true' if has_regression else 'false'}\n")


if __name__ == "__main__":
    main()
