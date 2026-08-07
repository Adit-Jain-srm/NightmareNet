from unittest import mock

from nightmarenet.pipeline import Pipeline


@mock.patch("nightmarenet.pipeline.TrainPhase")
def test_pipeline_train_phase_with_distributed_mode(mock_train_phase):
    """
    Ensure the pipeline passes through distributed mode correctly
    and executes TrainPhase successfully.
    """

    mock_result = mock.Mock()
    mock_result.success = True
    mock_result.data = {"history": []}

    mock_train_phase.return_value.execute.return_value = mock_result

    config = {
        "training": {
            "distributed": True,
            "num_cycles": 1,
        }
    }

    pipeline = Pipeline(
        config=config,
        distributed="auto",
    )

    trainer = mock.Mock()
    pipeline._context.trainer = trainer

    history = pipeline.train()

    assert history == []

    mock_train_phase.assert_called_once()

    mock_train_phase.return_value.execute.assert_called_once_with(pipeline._context)
