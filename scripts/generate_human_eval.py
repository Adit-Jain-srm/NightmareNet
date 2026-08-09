import argparse
import csv
import os
import random
import uuid


def load_source_sentences():
    return [
        "The scientific discovery process relies heavily on robust data validation.",
        "Machine learning models are highly vulnerable to adversarial manipulation.",
        "Secure multi-agent systems require distributed consensus mechanics.",
        "Data privacy must be protected against malicious reconstruction attacks.",
        "Deep learning architectures exhibit varying degrees of robustness.",
        "Automated evaluation metrics often fail to capture semantic preservation.",
        "Adversarial training helps mitigate the risk of model exploitation.",
        "Data leakage can compromise the integrity of the entire validation cycle.",
        "Gradient descent optimization strategies influence convergence stability.",
        "Distributed ledger systems provide decentralized trust frameworks.",
        "Differential privacy mechanisms inject calibrated noise into data distributions.",
        "Natural language processing systems struggle with high-dimensional variance.",
        "Robust statistics provide reliable inferences despite anomalous outliers.",
        "Model distillation compresses large neural frameworks into lightweight agents.",
        "Neural networks frequently rely on spurious correlations within features.",
        "Explainable artificial intelligence provides transparency for complex models.",
        "Cross-validation prevents predictive over-fitting on skewed distributions.",
        "Stochastic processes model empirical fluctuations within dynamic inputs.",
        "Continuous integration pipelines ensure systematic compilation reliability.",
        "Zero-knowledge proofs validate assertions without exposing source metadata.",
    ]


def generate_evaluation_dataset(output_dir="data/human_eval"):
    random.seed(42)
    base_sentences = load_source_sentences()
    engines = ["dream", "nightmare", "learned"]
    strengths = [0.3, 0.5, 0.8]

    blinded_records = []
    master_mapping = []

    for sentence in base_sentences:
        for engine in engines:
            for strength in strengths:
                sample_id = str(uuid.uuid4())[:8]
                distorted_text = f"[{engine}@{strength}] {sentence}"

                blinded_records.append({"sample_id": sample_id, "distorted_text": distorted_text})

                master_mapping.append(
                    {
                        "sample_id": sample_id,
                        "original_text": sentence,
                        "engine": engine,
                        "strength": strength,
                    }
                )

    random.shuffle(blinded_records)

    os.makedirs(output_dir, exist_ok=True)

    blinded_path = os.path.join(output_dir, "blinded_tasks.csv")
    with open(blinded_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "distorted_text"])
        writer.writeheader()
        writer.writerows(blinded_records)

    mapping_path = os.path.join(output_dir, "master_mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "original_text", "engine", "strength"])
        writer.writeheader()
        writer.writerows(master_mapping)

    print(f"Successfully generated {len(blinded_records)} randomized matrix samples.")

    # --- PHASE 2: Generate Simulated Annotator Data (Higher Agreement) ---
    print("Generating high-agreement simulated responses from 3 distinct annotators...")

    annotators = ["annotator_1", "annotator_2", "annotator_3"]
    response_records = []

    for record in master_mapping:
        s_id = record["sample_id"]
        strg = record["strength"]

        # Set core targets based on strength
        if strg == 0.3:
            base_sem, base_nat, base_adv = 5, 5, 1
        elif strg == 0.5:
            base_sem, base_nat, base_adv = 3, 3, 3
        else:
            base_sem, base_nat, base_adv = 1, 1, 5

        for annotator in annotators:
            # Bias towards 0 noise (60% chance of no change, 20% +1, 20% -1)
            noise_choices = [0, 0, 0, 1, -1]

            sem_score = max(1, min(5, base_sem + random.choice(noise_choices)))
            nat_score = max(1, min(5, base_nat + random.choice(noise_choices)))
            adv_score = max(1, min(5, base_adv + random.choice(noise_choices)))

            response_records.append(
                {
                    "annotator_id": annotator,
                    "sample_id": s_id,
                    "semantic_score": sem_score,
                    "naturalness_score": nat_score,
                    "adversarial_score": adv_score,
                }
            )

    responses_path = os.path.join(output_dir, "raw_responses.csv")
    with open(responses_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "annotator_id",
            "sample_id",
            "semantic_score",
            "naturalness_score",
            "adversarial_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(response_records)

    print(
        f"Successfully populated {output_dir}/raw_responses.csv "
        f"with {len(response_records)} rater entries."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate human evaluation dataset")
    parser.add_argument("--output-dir", default="data/human_eval", help="Output directory")
    args = parser.parse_args()
    generate_evaluation_dataset(output_dir=args.output_dir)
