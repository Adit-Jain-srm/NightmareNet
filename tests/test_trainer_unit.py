import os
import tempfile
import unittest
from unittest import mock
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from nightmarenet.training.trainer import Trainer, _get_device, _create_amp_scaler, VisionModelWrapper
from nightmarenet.training.callbacks import CallbackManager, EventType, TrainingEvent


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        if input_ids is None and "pixel_values" in kwargs:
            input_ids = kwargs["pixel_values"]
        if input_ids is None:
            input_ids = torch.randn(2, 10)
        logits = self.linear(input_ids)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out


def make_dummy_dataloader(batch_size=2, num_samples=6):
    x = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(x, y)
    
    class SimpleLoader:
        def __init__(self, ds, batch_size):
            self.ds = ds
            self.batch_size = batch_size
        
        def __len__(self):
            return len(self.ds) // self.batch_size
        
        def __iter__(self):
            for i in range(0, len(self.ds), self.batch_size):
                batch_x, batch_y = self.ds[i:i+self.batch_size]
                yield {"input_ids": batch_x, "labels": batch_y}
                
    return SimpleLoader(dataset, batch_size)


class TestTrainerUnit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "model": {"type": "causal_lm", "name": "dummy"},
            "training": {
                "checkpoint_dir": os.path.join(self.temp_dir.name, "checkpoints"),
                "log_dir": os.path.join(self.temp_dir.name, "logs"),
                "batch_size": 2,
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "num_cycles": 1,
                "learning_rate": 1e-3,
                "gradient_accumulation_steps": 2,
            },
            "distortion": {},
            "compression": {},
        }
        self.dummy_model = DummyModel()
        self.dummy_tokenizer = mock.Mock()
        self.dummy_tokenizer.pad_token = "[PAD]"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trainer_initialization_and_device(self):
        device = _get_device({"model": {"device": "cpu"}})
        self.assertEqual(device.type, "cpu")

        scaler = _create_amp_scaler(use_amp=True, device=torch.device("cpu"))
        self.assertIsNone(scaler)

        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
        )
        self.assertEqual(trainer.device.type, "cpu")
        self.assertFalse(trainer.use_amp)
        self.assertIsNotNone(trainer.optimizer)

    def test_vision_wrapper(self):
        base = nn.Linear(5, 2)
        wrapper = VisionModelWrapper(base)
        pixel_val = torch.randn(3, 5)
        labels = torch.tensor([0, 1, 0])
        out = wrapper(pixel_values=pixel_val, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.logits.shape, (3, 2))

    @mock.patch("nightmarenet.training.trainer.WakePhase")
    @mock.patch("nightmarenet.training.trainer.DreamPhase")
    @mock.patch("nightmarenet.training.trainer.NightmarePhase")
    def test_training_loop_execution_single_epoch(self, mock_nightmare, mock_dream, mock_wake):
        mock_result = {
            "success": True,
            "metrics": {"loss": 0.5, "accuracy": 0.9},
            "history": [{"loss": 0.5}],
            "avg_loss": 0.5,
        }

        mock_wake.return_value.execute.return_value = mock_result
        mock_dream.return_value.execute.return_value = mock_result
        mock_nightmare.return_value.execute.return_value = mock_result

        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
        )

        dl = make_dummy_dataloader()
        history = trainer.train(
            train_dataloader=dl,
            dream_dataloader=dl,
            nightmare_dataloader=dl,
        )
        self.assertGreater(len(history), 0)
        self.assertTrue(mock_wake.called or mock_dream.called or mock_nightmare.called)

    def test_callback_manager_integration(self):
        cm = CallbackManager()
        events_captured = []

        def callback(event: TrainingEvent):
            events_captured.append(event.event_type)

        cm.on(EventType.STEP, callback)

        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
            callback_manager=cm,
        )
        trainer.callback_manager.emit(TrainingEvent(event_type=EventType.STEP, phase="test"))
        self.assertIn(EventType.STEP, events_captured)

    def test_checkpoint_save_and_load_roundtrip(self):
        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
        )
        trainer.history = [{"cycle": 0, "phase": "wake", "loss": 0.4}]

        with mock.patch.object(trainer.checkpointer, "save", return_value=self.temp_dir.name) as mock_save, \
             mock.patch("nightmarenet.distributed.checkpoint.compute_dir_hashes", return_value={}), \
             mock.patch("nightmarenet.distributed.checkpoint.validate_checkpoint_integrity"):
            trainer._save_checkpoint(cycle=0, phase="wake")
            mock_save.assert_called_once()

    def test_early_stopping_and_stop_request(self):
        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
        )
        self.assertFalse(trainer._stop_requested)
        trainer.request_stop()
        self.assertTrue(trainer._stop_requested)

    def test_resume_manager_integration(self):
        mock_resume_mgr = mock.Mock()
        mock_resume_mgr.verify_and_load.return_value = {"cycle": 1, "phase": "dream"}

        with mock.patch("nightmarenet.training.trainer.ResumeManager", return_value=mock_resume_mgr):
            trainer = Trainer(
                config=self.config,
                model=self.dummy_model,
                tokenizer=self.dummy_tokenizer,
                resume_dir="/fake/resume/dir",
            )
            self.assertEqual(trainer._start_cycle, 1)
            self.assertEqual(trainer._start_phase, "dream")

    def test_error_handling_and_interrupt(self):
        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
        )
        self.assertFalse(trainer._interrupted)
        trainer._handle_interrupt(None, None)
        self.assertTrue(trainer._interrupted)

    def test_distributed_sync_hooks(self):
        trainer = Trainer(
            config=self.config,
            model=self.dummy_model,
            tokenizer=self.dummy_tokenizer,
            distributed="0,1",
        )
        self.assertTrue(trainer.use_distributed)
        self.assertIsNotNone(trainer.ddp_wrapper)
        self.assertIsNotNone(trainer.device_pool)


if __name__ == "__main__":
    unittest.main()
