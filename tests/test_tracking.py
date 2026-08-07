import unittest
from unittest import mock
from nightmarenet.utils.tracking import ExperimentTracker, create_tracker_from_config


class TestTrackingUnit(unittest.TestCase):
    def test_create_tracker_from_config(self):
        config_disabled = {"tracking": {"backend": "none"}}
        tracker = create_tracker_from_config(config_disabled)
        self.assertEqual(tracker.backend, "none")

        config_mlflow = {"tracking": {"backend": "mlflow", "experiment": "test_exp"}}
        with mock.patch.dict("sys.modules", {"mlflow": mock.MagicMock()}):
            tracker_mlflow = create_tracker_from_config(config_mlflow)
            self.assertEqual(tracker_mlflow.backend, "mlflow")

    def test_metric_recording_and_lifecycle(self):
        tracker = ExperimentTracker(backend="none")
        tracker.log_config({"learning_rate": 1e-4})
        self.assertEqual(tracker.lineage["config"]["learning_rate"], 1e-4)

        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.4}, step=2)

        tracker.log_artifact("checkpoint.pt")
        tracker.finish()

    def test_error_resilience_on_invalid_backend(self):
        tracker = ExperimentTracker(backend="non_existent_backend")
        self.assertEqual(tracker.backend, "none")


if __name__ == "__main__":
    unittest.main()
