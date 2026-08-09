"""Integration test for the computer vision sleep-cycle pipeline."""

from nightmarenet.pipeline import Pipeline, PipelineStatus


def test_full_vision_pipeline_cycle():
    config = {
        "model": {
            "name": "resnet18",
            "type": "image_classification",
            "num_labels": 10,
            "device": "cpu",
        },
        "dataset": {
            "name": "cifar10",
            "max_samples": 10,
        },
        "training": {
            "wake_epochs": 1,
            "dream_epochs": 1,
            "nightmare_epochs": 1,
            "compress_epochs": 1,
            "num_cycles": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "save_every_phase": False,
            "checkpoint_dir": "checkpoints_vision",
            "log_dir": "logs_vision",
        },
        "distortion": {
            "dream_strength": 0.25,
            "nightmare_strength": 0.75,
        },
        "compression": {
            "pruning_ratio": 0.1,
            "pruning_method": "magnitude",
        },
        "evaluation": {
            "metrics": ["classification", "robustness"],
            "strengths": [0.2, 0.6],
        },
        "tracking": {"backend": "none"},
        "seed": 42,
    }

    pipe = Pipeline(config)

    # 1. Ingest (mock CIFAR-10 download to force fallback to FakeData immediately)
    from unittest.mock import patch

    side_effect = Exception("Mock network download failure")
    with patch("torchvision.datasets.CIFAR10", side_effect=side_effect):
        pipe.ingest()
    assert pipe.metrics.status == PipelineStatus.INGESTING

    # 2. Prepare
    pipe.prepare()
    assert pipe.metrics.status == PipelineStatus.PREPARING

    # 3. Train
    pipe.train()

    # 4. Evaluate
    pipe.evaluate()
    assert pipe.metrics.status == PipelineStatus.EVALUATING
    assert pipe.metrics.error is None
    assert pipe.metrics.report_md is not None

    # Check that evaluation comparison results exist and are correct
    assert pipe._context.comparison is not None
    comparison = pipe._context.comparison
    assert "metrics" in comparison
    assert "classification" in comparison["metrics"]
    assert "robustness" in comparison["metrics"]
