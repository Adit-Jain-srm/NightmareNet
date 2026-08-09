"""Expanded test coverage for nightmarenet.hub (push/pull, model cards, hash checks).

Complements tests/test_hub.py without duplicating its scenarios. All HTTP/network
access to the HuggingFace Hub is mocked; no real registry, network, or GPU access
is required to run this suite.
"""

import builtins
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nightmarenet.hub.core import pull_model, push_model
from nightmarenet.hub.model_card import generate_model_card
from nightmarenet.hub.utils import require_hf_hub

# ---------------------------------------------------------------------------
# core.py — push_model / pull_model
# ---------------------------------------------------------------------------


@patch("huggingface_hub.HfApi")
@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_push_model_upload_payload_targets_correct_repo(mock_hf_api, tmp_path):
    """Verify push_model routes the correct repo_id/repo_type/folder_path to the upload call."""
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance

    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    push_model(model_dir=str(model_dir), repo_id="test-user/payload-check")

    _, upload_kwargs = mock_api_instance.upload_folder.call_args
    assert upload_kwargs["repo_id"] == "test-user/payload-check"
    assert upload_kwargs["repo_type"] == "model"
    assert upload_kwargs["folder_path"] == str(model_dir)


@patch("huggingface_hub.HfApi")
@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_push_model_network_error_propagates(mock_hf_api, tmp_path):
    """Verify a network failure during upload raises rather than being silently swallowed."""
    mock_api_instance = MagicMock()
    mock_api_instance.upload_folder.side_effect = ConnectionError("simulated network failure")
    mock_hf_api.return_value = mock_api_instance

    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with pytest.raises(ConnectionError, match="simulated network failure"):
        push_model(model_dir=str(model_dir), repo_id="test-user/network-fail")

    # The model card should still have been written locally before the network call failed.
    assert (model_dir / "README.md").exists()
    mock_api_instance.create_repo.assert_called_once()


@patch.dict("os.environ", {}, clear=True)
def test_push_model_missing_token_raises_runtime_error(tmp_path):
    """Verify push_model fails fast with a clear error when HF_TOKEN is unset."""
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_model(model_dir=str(model_dir), repo_id="test-user/no-token")


@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_push_model_missing_local_dir_raises(tmp_path):
    """Verify push_model raises FileNotFoundError for a nonexistent model directory."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="not found"):
        push_model(model_dir=str(missing_dir), repo_id="test-user/missing-dir")


@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_push_model_invalid_metadata_yaml_type_raises(tmp_path):
    """Verify push_model rejects metadata YAML that isn't a mapping (e.g. a list)."""
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    metadata_file = tmp_path / "metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.safe_dump(["not", "a", "mapping"], f)

    with pytest.raises(TypeError, match="mapping"):
        push_model(
            model_dir=str(model_dir),
            repo_id="test-user/bad-metadata",
            metadata_path=str(metadata_file),
        )


@patch("huggingface_hub.snapshot_download")
def test_pull_model_download_network_error_propagates(mock_snapshot, tmp_path):
    """Verify pull_model surfaces network failures instead of hiding them."""
    mock_snapshot.side_effect = ConnectionError("simulated network failure")
    target_dir = tmp_path / "download_target"

    with pytest.raises(ConnectionError, match="simulated network failure"):
        pull_model(repo_id="test-org/network-fail", target_dir=str(target_dir))

    # Target directory creation happens before the download attempt.
    assert target_dir.exists()


@patch("huggingface_hub.snapshot_download")
def test_pull_model_corrupted_download_hash_mismatch_propagates(mock_snapshot, tmp_path):
    """Verify a hash-mismatch/corrupted-download error from the downloader is not swallowed.

    huggingface_hub's snapshot_download performs its own file-integrity verification and
    raises on checksum mismatches; pull_model must propagate that failure rather than
    reporting a successful download.
    """
    mock_snapshot.side_effect = OSError("Consistency check failed: file hash mismatch")
    target_dir = tmp_path / "download_target"

    with pytest.raises(OSError, match="hash mismatch"):
        pull_model(repo_id="test-org/corrupted-weights", target_dir=str(target_dir))


@patch("huggingface_hub.snapshot_download")
@patch.dict("os.environ", {"HF_TOKEN": "mock_token_for_testing"})
def test_pull_model_forwards_hf_token(mock_snapshot, tmp_path):
    """Verify pull_model forwards HF_TOKEN from the environment to snapshot_download."""
    target_dir = tmp_path / "download_target"
    pull_model(repo_id="test-org/private-weights", target_dir=str(target_dir))

    mock_snapshot.assert_called_once_with(
        repo_id="test-org/private-weights",
        local_dir=str(target_dir),
        token="mock_token_for_testing",
    )


# ---------------------------------------------------------------------------
# model_card.py — generate_model_card
# ---------------------------------------------------------------------------


def _split_frontmatter(card_content: str):
    """Split rendered card markdown into (frontmatter_dict, body)."""
    assert card_content.startswith("---\n")
    _, frontmatter, body = card_content.split("---\n", 2)
    return yaml.safe_load(frontmatter), body


def test_generate_model_card_frontmatter_is_valid_yaml():
    """Verify the card's YAML frontmatter parses cleanly and the body follows it."""
    metadata = {
        "robustness_score": 0.75,
        "cycle_count": 3,
        "distortion_families": ["text", "semantic", "visual"],
    }
    card_content = generate_model_card("test-org/valid-frontmatter", metadata)
    frontmatter, body = _split_frontmatter(card_content)

    assert frontmatter["tags"] == ["nightmarenet", "robustness", "adversarial-defense"]
    assert "# NightmareNet Hardened Model" in body


def test_generate_model_card_contains_expected_sections():
    """Verify the rendered card includes the documented structural sections."""
    card_content = generate_model_card(
        "test-org/sectioned-model",
        {"robustness_score": 0.6, "cycle_count": 5, "distortion_families": ["text"]},
    )
    assert "## Model Training & Resilience Profile" in card_content
    assert "## Hardware Information" in card_content
    assert "## Reproducibility Metadata Configuration" in card_content


def test_generate_model_card_missing_optional_fields_uses_defaults():
    """Verify generate_model_card degrades gracefully when metadata is empty."""
    card_content = generate_model_card("test-org/no-metadata", {})
    frontmatter, body = _split_frontmatter(card_content)

    # robustness_score defaults to 0.0 for the ModelCardData metric block.
    assert frontmatter["model_index"][0]["results"][0]["metrics"][0]["value"] == 0.0
    # Missing cycle_count/distortion_families render as human-readable placeholders.
    assert "Cycle count: N/A" in body
    assert "**Distortion Vectors Defended:** None" in body
    assert "config_yaml" not in body  # template placeholder must be substituted, not literal


def test_generate_model_card_special_characters_no_injection():
    """Verify special/markdown-control characters in the repo id don't corrupt card structure."""
    repo_id = "org/model](javascript:alert(1))<script>alert(1)</script>"
    card_content = generate_model_card(repo_id, {})

    # The frontmatter must still be well-formed YAML despite the hostile repo_id.
    frontmatter, body = _split_frontmatter(card_content)
    assert frontmatter["tags"] == ["nightmarenet", "robustness", "adversarial-defense"]

    # The repo id is rendered verbatim inside an inline code span, not as a live markdown link.
    assert f"`{repo_id}`" in body
    assert "](javascript:" not in body.split(f"`{repo_id}`")[0]


def test_generate_model_card_null_config_yaml_block_is_placeholder():
    """Verify an empty/absent config renders a safe empty-object YAML block, not a crash."""
    card_content = generate_model_card("test-org/empty-config", {"config": {}})
    assert "```yaml\n{}\n```" in card_content


# ---------------------------------------------------------------------------
# utils.py — require_hf_hub
# ---------------------------------------------------------------------------


def test_require_hf_hub_passthrough_when_installed():
    """Verify the decorator calls through with args/kwargs/return value intact when available."""

    @require_hf_hub
    def dummy(a, b, keyword=None):
        return a + b, keyword

    assert dummy(1, 2, keyword="ok") == (3, "ok")


def test_require_hf_hub_raises_clear_import_error_when_missing(monkeypatch):
    """Verify a clear, actionable ImportError is raised when huggingface_hub is unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    @require_hf_hub
    def dummy():
        return "should not run"

    with pytest.raises(ImportError, match="pip install huggingface_hub"):
        dummy()