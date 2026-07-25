"""TextAttack integration for adversarial robustness evaluation.

Provides a wrapper around TextAttack's attack recipes (TextFooler,
BERTAttack, TextBugger, PWWS) for evaluating NightmareNet-trained models.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

ATTACK_RECIPES = {
    "textfooler": "TextFoolerJin2019",
    "bertattack": "BERTAttackLi2020",
    "textbugger": "TextBuggerLi2018",
    "pwws": "PWWSRen2019",
}


def _check_textattack_available() -> None:
    """Raise ImportError with install instructions if textattack is missing."""
    try:
        import textattack  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "textattack is not installed. Install with: "
            "pip install 'nightmarenet[attacks]'"
        ) from exc


def get_recipe(attack_name: str, model_wrapper: Any) -> Any:
    """Get an initialized attack recipe.

    Args:
        attack_name: One of textfooler, bertattack, textbugger, pwws.
        model_wrapper: A TextAttack HuggingFaceModelWrapper instance.

    Returns:
        Initialized TextAttack Attack object.

    Raises:
        ImportError: If textattack is not installed.
        ValueError: If attack_name is not supported.
    """
    _check_textattack_available()

    name = attack_name.lower()
    if name not in ATTACK_RECIPES:
        raise ValueError(
            f"Unsupported attack: {attack_name!r}. "
            f"Supported: {list(ATTACK_RECIPES.keys())}"
        )

    from textattack.attack_recipes import (
        BERTAttackLi2020,
        PWWSRen2019,
        TextBuggerLi2018,
        TextFoolerJin2019,
    )

    recipes = {
        "textfooler": TextFoolerJin2019,
        "bertattack": BERTAttackLi2020,
        "textbugger": TextBuggerLi2018,
        "pwws": PWWSRen2019,
    }
    return recipes[name].build(model_wrapper)


def run_textattack_evaluation(
    model: Any,
    tokenizer: Any,
    dataset_name: str = "sst2",
    attack_names: Optional[list[str]] = None,
    num_examples: int = 200,
    device: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Run TextAttack adversarial evaluation.

    Args:
        model: HuggingFace classification model.
        tokenizer: HuggingFace tokenizer.
        dataset_name: Dataset to evaluate on (default: sst2).
        attack_names: List of attacks to run. Defaults to ["textfooler"].
        num_examples: Number of examples to evaluate.
        device: Device to run on (e.g. "cuda", "cpu"). None = auto.

    Returns:
        Dict mapping attack name to results dict with keys:
        attack_success_rate, successful_attacks, failed_attacks,
        skipped_attacks, avg_queries, avg_perturbation_pct, elapsed_time.
        On failure, the dict contains an "error" key instead.

    Raises:
        ImportError: If textattack is not installed.
    """
    _check_textattack_available()

    import textattack
    from textattack import AttackArgs, Attacker
    from textattack.datasets import HuggingFaceDataset
    from textattack.models.wrappers import HuggingFaceModelWrapper

    if not attack_names:
        attack_names = ["textfooler"]

    logger.info(
        "Running TextAttack evaluation with %s on %d examples of %s",
        attack_names,
        num_examples,
        dataset_name,
    )

    if device is not None:
        model = model.to(device)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    if dataset_name.lower() == "sst2":
        dataset = HuggingFaceDataset("glue", "sst2", split="validation")
    else:
        dataset = HuggingFaceDataset(dataset_name, split="validation")

    results: dict[str, dict[str, Any]] = {}

    for attack_name in attack_names:
        logger.info("Running attack: %s", attack_name)
        try:
            attack = get_recipe(attack_name, model_wrapper)

            attack_args = AttackArgs(
                num_examples=num_examples,
                log_to_csv=None,
                log_to_txt=None,
                disable_stdout=True,
                parallel=False,
            )

            attacker = Attacker(attack, dataset, attack_args)

            start_time = time.time()
            attack_results = attacker.attack_dataset()
            elapsed_time = time.time() - start_time

            successful_attacks = sum(
                1 for r in attack_results
                if isinstance(r, textattack.attack_results.SuccessfulAttackResult)
            )
            failed_attacks = sum(
                1 for r in attack_results
                if isinstance(r, textattack.attack_results.FailedAttackResult)
            )
            skipped_attacks = sum(
                1 for r in attack_results
                if isinstance(r, textattack.attack_results.SkippedAttackResult)
            )

            total_attempted = successful_attacks + failed_attacks
            asr = (
                (successful_attacks / total_attempted) * 100
                if total_attempted > 0
                else 0.0
            )

            non_skipped = [
                r for r in attack_results
                if not isinstance(r, textattack.attack_results.SkippedAttackResult)
            ]
            avg_queries = (
                sum(r.num_queries for r in non_skipped) / len(non_skipped)
                if non_skipped
                else 0.0
            )

            perturbation_pcts: list[float] = []
            for r in attack_results:
                if isinstance(r, textattack.attack_results.SuccessfulAttackResult):
                    original_words = r.original_result.attacked_text.words
                    perturbed_words = r.perturbed_result.attacked_text.words
                    diff_words = sum(
                        1 for w1, w2 in zip(original_words, perturbed_words)
                        if w1 != w2
                    )
                    diff_words += abs(len(original_words) - len(perturbed_words))
                    pct = diff_words / max(len(original_words), 1) * 100
                    perturbation_pcts.append(pct)

            avg_perturbation = (
                sum(perturbation_pcts) / len(perturbation_pcts)
                if perturbation_pcts
                else 0.0
            )

            results[attack_name] = {
                "attack_success_rate": asr,
                "successful_attacks": successful_attacks,
                "failed_attacks": failed_attacks,
                "skipped_attacks": skipped_attacks,
                "avg_queries": avg_queries,
                "avg_perturbation_pct": avg_perturbation,
                "elapsed_time": elapsed_time,
            }

        except Exception as exc:
            logger.error("Failed to run attack %s: %s", attack_name, exc)
            results[attack_name] = {"error": str(exc)}

    return results


def format_comparison_table(
    results: dict[str, dict[str, Any]], dataset_name: str = "sst2"
) -> str:
    """Format results as a markdown-like comparison table.

    Args:
        results: Dict from run_textattack_evaluation.
        dataset_name: Dataset name for baseline comparison.

    Returns:
        Formatted string table.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("   TEXTATTACK ROBUSTNESS EVALUATION (NightmareNet)   ")
    lines.append("=" * 60)
    lines.append(f"Dataset: {dataset_name}")
    lines.append("")
    lines.append(
        f"{'Attack':<15} | {'ASR (%)':<10} | {'Perturb (%)':<12} | "
        f"{'Queries':<10} | {'Baseline ASR':<15}"
    )
    lines.append("-" * 75)

    baselines = {
        ("textfooler", "sst2"): "89.2%",
        ("bertattack", "sst2"): "87-92%",
    }

    for attack, res in results.items():
        if "error" in res:
            lines.append(
                f"{attack.upper():<15} | {'ERROR':<10} | {'-':<12} | "
                f"{'-':<10} | {'-':<15}"
            )
            continue

        asr = f"{res['attack_success_rate']:.1f}"
        pert = f"{res['avg_perturbation_pct']:.1f}"
        queries = f"{res['avg_queries']:.1f}"
        baseline = baselines.get((attack, dataset_name), "N/A")
        lines.append(
            f"{attack.upper():<15} | {asr:<10} | {pert:<12} | "
            f"{queries:<10} | {baseline:<15}"
        )

    lines.append("-" * 75)
    lines.append("Note: Baseline ASR is vs vanilla BERT on SST-2.")
    lines.append("      Lower ASR = stronger robustness.")
    lines.append("=" * 60)
    return "\n".join(lines)
