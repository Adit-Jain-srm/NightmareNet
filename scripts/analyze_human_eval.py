import argparse
import csv
import os

import numpy as np


def load_data(input_dir="data/human_eval"):
    """Loads raw responses and master mapping records into memory."""
    responses = []
    with open(os.path.join(input_dir, "raw_responses.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            responses.append(
                {
                    "annotator_id": row["annotator_id"],
                    "sample_id": row["sample_id"],
                    "semantic_score": int(row["semantic_score"]),
                    "naturalness_score": int(row["naturalness_score"]),
                    "adversarial_score": int(row["adversarial_score"]),
                }
            )

    mapping = {}
    with open(os.path.join(input_dir, "master_mapping.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["sample_id"]] = {
                "engine": row["engine"],
                "strength": float(row["strength"]),
            }

    return responses, mapping


def compute_krippendorff_alpha(sample_ratings, n_categories=5):
    """
    Computes Krippendorff's Alpha with an Ordinal metric constraint.
    Expects sample_ratings to map sample_ids to a list of rater integers (1-5).
    """
    # Build reliability matrix (n_samples x n_annotators)
    samples = list(sample_ratings.keys())
    n_samples = len(samples)
    n_annotators = len(sample_ratings[samples[0]])

    reliability_matrix = np.zeros((n_samples, n_annotators))
    for i, s_id in enumerate(samples):
        reliability_matrix[i, :] = sample_ratings[s_id]

    # Coincidence matrix calculation setup
    # Track distributions of paired annotations
    coincidence = np.zeros((n_categories, n_categories))

    for row in reliability_matrix:
        for i in range(n_annotators):
            for j in range(n_annotators):
                if i != j:
                    v1 = int(row[i]) - 1
                    v2 = int(row[j]) - 1
                    coincidence[v1, v2] += 1.0 / (n_annotators - 1)

    total_pairs = np.sum(coincidence)

    # Calculate linear/ordinal distance weights
    # Distance delta(g, k) = (g - k)^2 for ordinal metrics
    observed_disagreement = 0.0
    for g in range(n_categories):
        for k in range(n_categories):
            observed_disagreement += coincidence[g, k] * ((g - k) ** 2)

    # Expected agreement distributions by chance
    n_k = np.sum(coincidence, axis=1)
    expected_disagreement = 0.0
    for g in range(n_categories):
        for k in range(n_categories):
            expected_disagreement += n_k[g] * n_k[k] * ((g - k) ** 2)

    expected_disagreement /= max(total_pairs - 1, 1)

    if expected_disagreement == 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def execute_analysis(input_dir="data/human_eval"):
    responses, mapping = load_data(input_dir)

    condition_metrics = {}
    semantic_ratings = {}
    naturalness_ratings = {}
    adversarial_ratings = {}

    for resp in responses:
        s_id = resp["sample_id"]
        meta = mapping[s_id]
        cond_key = (meta["engine"], meta["strength"])

        if cond_key not in condition_metrics:
            condition_metrics[cond_key] = {"semantic": [], "naturalness": [], "adversarial": []}

        condition_metrics[cond_key]["semantic"].append(resp["semantic_score"])
        condition_metrics[cond_key]["naturalness"].append(resp["naturalness_score"])
        condition_metrics[cond_key]["adversarial"].append(resp["adversarial_score"])

        semantic_ratings.setdefault(s_id, []).append(resp["semantic_score"])
        naturalness_ratings.setdefault(s_id, []).append(resp["naturalness_score"])
        adversarial_ratings.setdefault(s_id, []).append(resp["adversarial_score"])

    print("=== INTER-ANNOTATOR AGREEMENT METRICS (KRIPPENDORFF'S ALPHA) ===")
    alpha_sem = compute_krippendorff_alpha(semantic_ratings)
    alpha_nat = compute_krippendorff_alpha(naturalness_ratings)
    alpha_adv = compute_krippendorff_alpha(adversarial_ratings)

    print(f"Krippendorff's Alpha (Semantic Preservation): {alpha_sem:.4f}")
    print(f"Krippendorff's Alpha (Naturalness Quality):   {alpha_nat:.4f}")
    print(f"Krippendorff's Alpha (Adversarial Strength):  {alpha_adv:.4f}")

    kappa_targets = [
        ("Semantic", alpha_sem),
        ("Naturalness", alpha_nat),
        ("Adversarial", alpha_adv),
    ]
    for name, val in kappa_targets:
        status = "PASSED" if val > 0.4 else "FAILED"
        print(f"  -> Verification [{name} > 0.4]: {status}")

    print("\n=== MEAN TIER PERFORMANCE SUMMARY ===")
    print(f"{'Engine':<12} | {'Strength':<8} | {'Semantic':<8} | {'Natural':<8} | {'Advers':<8}")
    print("-" * 55)

    for (engine, strength), metrics in sorted(condition_metrics.items()):
        m_sem = np.mean(metrics["semantic"])
        m_nat = np.mean(metrics["naturalness"])
        m_adv = np.mean(metrics["adversarial"])
        print(f"{engine:<12} | {strength:<8} | {m_sem:<8.2f} | {m_nat:<8.2f} | {m_adv:<8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze human evaluation results")
    parser.add_argument("--input-dir", default="data/human_eval", help="Input directory")
    args = parser.parse_args()
    execute_analysis(input_dir=args.input_dir)
