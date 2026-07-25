import os
import shutil
import tempfile

import pytest

from nightmarenet.pipeline import Pipeline


@pytest.mark.slow
def test_4_phase_training_cycle_e2e():
    """
    End-to-end integration test running the full 4-phase training cycle
    on a real model with a minimal dataset to ensure the system works end-to-end.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        config = {
            "seed": 42,
            "dataset": {
                "name": "glue",
                "config": "sst2",
                "text_column": "sentence",
                "label_column": "label",
                "max_samples": 50,
            },
            "model": {
                "name": "distilbert-base-uncased",
                "type": "masked_lm",
                "max_length": 128,
                "device": "cpu",
            },
            "training": {
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "num_cycles": 1,
                "batch_size": 8,
                "learning_rate": 3.0e-5,
                "output_dir": temp_dir,
            },
            "distortion": {
                "dream_strength": 0.2,
                "nightmare_strength": 0.5,
            },
            "compression": {
                "pruning_ratio": 0.1,
            },
            "evaluation": {
                "robustness_strengths": [0.1, 0.3],
                "eval_split_ratio": 0.2,
            },
            "tracking": {
                "backend": "none",
            },
        }

        pipeline = Pipeline(config)

        # Run the full pipeline
        comparison = pipeline.run(hf_dataset="glue", hf_subset="sst2", export_dir=temp_dir)

        # 1. Assert evaluation comparison dict has expected keys
        assert comparison is not None, "Evaluation comparison should not be None"
        assert "metrics" in comparison, f"Missing metrics key: {comparison}"

        # 2. Assert robustness_score > 0
        metrics = comparison.get("metrics", {})
        assert "robustness" in metrics, f"Robustness metrics missing. Got: {list(metrics.keys())}"
        robustness = metrics["robustness"]

        trained_robustness = robustness.get("trained", {})
        auc_robustness = trained_robustness.get("auc_robustness")
        assert auc_robustness is not None, (
            f"AUC robustness missing. Trained stats: {trained_robustness}"
        )
        assert auc_robustness > 0.0, f"Robustness score should be > 0, got {auc_robustness}"

        # 3. Assert model checkpoint saved at expected path
        # Pipeline.export() saves to export_dir which is temp_dir
        assert os.path.exists(os.path.join(temp_dir, "model.safetensors")) or os.path.exists(
            os.path.join(temp_dir, "pytorch_model.bin")
        ), "Model weights file not found in export directory"
        assert os.path.exists(os.path.join(temp_dir, "config.json")), (
            "Model config.json not found in export directory"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
