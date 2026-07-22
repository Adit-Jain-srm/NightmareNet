from unittest.mock import MagicMock, patch

import yaml

from nightmarenet.hub.core import pull_model, push_model
from nightmarenet.hub.model_card import generate_model_card


def test_generate_model_card_formatting():
    """Verify that the model card builds the correct YAML frontmatter and markdown layout."""
    repo_id = "test-org/robust-model"
    metadata = {
        "robustness_score": 0.8542,
        "cycle_count": 4,
        "distortion_families": ["text", "semantic"],
        "config": {"model": {"name": "distilbert-base-uncased"}},
    }
    card_content = generate_model_card(repo_id, metadata)
    # Assert tag headers are present
    assert "nightmarenet" in card_content
    assert "robustness" in card_content
    assert "nightmarenet_cycle_count: 4" in card_content
    assert "**Robustness Score:** 0.8542" in card_content
    assert "distilbert-base-uncased" in card_content


@patch("huggingface_hub.HfApi")
@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_push_model_execution(mock_hf_api, tmp_path):
    """Verify push_model writes README.md and calls HfApi upload methods correctly."""
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance
    # Setup dummy local directory structure
    model_dir = tmp_path / "model_artifacts"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_text("dummy_weights")
    metadata_file = tmp_path / "metadata.yaml"
    dummy_meta = {"robustness_score": 0.91, "cycle_count": 2}
    with open(metadata_file, "w") as f:
        yaml.safe_dump(dummy_meta, f)
    # Execute the push
    push_model(
        model_dir=str(model_dir),
        repo_id="test-user/hardened-test",
        metadata_path=str(metadata_file),
    )
    # Verify local file generation and API calls
    assert (model_dir / "README.md").exists()
    mock_api_instance.create_repo.assert_called_once_with(
        repo_id="test-user/hardened-test", exist_ok=True, repo_type="model"
    )
    mock_api_instance.upload_folder.assert_called_once()


@patch("huggingface_hub.snapshot_download")
def test_pull_model_execution(mock_snapshot, tmp_path):
    """Verify pull_model creates target folders and routes repo arguments to snapshot downloader."""
    target_dir = tmp_path / "download_target"
    pull_model(repo_id="test-org/public-weights", target_dir=str(target_dir))
    assert target_dir.exists()
    mock_snapshot.assert_called_once_with(
        repo_id="test-org/public-weights", local_dir=str(target_dir), token=None
    )
