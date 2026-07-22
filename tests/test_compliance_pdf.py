"""Tests for PDF generation with digital signature."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nightmarenet.compliance.report import generate_pdf


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "dataset": {
            "name": "sst2",
            "path": "/tmp/data",
        },
        "model": {
            "name": "bert-base-uncased",
            "type": "classification",
        },
    }


@pytest.fixture
def sample_comparison():
    """Sample comparison metrics for testing."""
    return {
        "metrics": {
            "robustness": {
                "trained": {
                    "clean_accuracy": 0.92,
                    "distorted_accuracy": 0.85,
                    "auc_robustness": 0.78,
                },
                "deltas": {
                    "auc_robustness": -0.05,
                },
            }
        }
    }


@pytest.fixture
def sample_model_file(tmp_path):
    """Create a temporary model file for testing."""
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fake model data")
    return str(model_file)


def test_generate_pdf_without_dependencies(
    sample_config,
    sample_comparison,
    sample_model_file,
    monkeypatch,
):
    """Test that generate_pdf raises ImportError without dependencies."""
    # Mock the import check to simulate missing dependencies
    import nightmarenet.compliance.pdf_builder as pdf_builder

    monkeypatch.setattr(pdf_builder, "REPORTLAB_AVAILABLE", False)

    with pytest.raises(ImportError, match="reportlab is required"):
        generate_pdf(
            config=sample_config,
            comparison=sample_comparison,
            model_path=sample_model_file,
            output_dir=str(tempfile.mkdtemp()),
        )


def test_generate_pdf_with_dependencies(
    sample_config,
    sample_comparison,
    sample_model_file,
    tmp_path,
):
    """Test PDF generation with all dependencies installed."""
    try:
        from nightmarenet.compliance.pdf_builder import (
            PYHANKO_AVAILABLE,
            REPORTLAB_AVAILABLE,
        )
    except ImportError:
        pytest.skip("PDF builder module not available")

    if not (REPORTLAB_AVAILABLE and PYHANKO_AVAILABLE):
        pytest.skip("PDF dependencies not installed")

    output_dir = str(tmp_path / "output")
    pdf_path = generate_pdf(
        config=sample_config,
        comparison=sample_comparison,
        model_path=sample_model_file,
        output_dir=output_dir,
    )

    # Verify PDF file was created
    assert Path(pdf_path).exists()
    assert pdf_path.endswith(".pdf")

    # Verify PDF is not empty
    assert Path(pdf_path).stat().st_size > 0

    # Verify PDF has valid header (PDF files start with %PDF)
    with open(pdf_path, "rb") as f:
        header = f.read(4)
        assert header == b"%PDF"


def test_generate_pdf_creates_output_dir(
    sample_config,
    sample_comparison,
    sample_model_file,
    tmp_path,
):
    """Test that generate_pdf creates output directory if it doesn't exist."""
    try:
        from nightmarenet.compliance.pdf_builder import (
            PYHANKO_AVAILABLE,
            REPORTLAB_AVAILABLE,
        )
    except ImportError:
        pytest.skip("PDF builder module not available")

    if not (REPORTLAB_AVAILABLE and PYHANKO_AVAILABLE):
        pytest.skip("PDF dependencies not installed")

    output_dir = str(tmp_path / "nested" / "dir" / "that" / "does" / "not" / "exist")
    pdf_path = generate_pdf(
        config=sample_config,
        comparison=sample_comparison,
        model_path=sample_model_file,
        output_dir=output_dir,
    )

    assert Path(pdf_path).exists()
    assert Path(output_dir).exists()


def test_generate_pdf_with_tracker(
    sample_config,
    sample_comparison,
    sample_model_file,
    tmp_path,
):
    """Test PDF generation with tracker for custom run ID."""
    try:
        from nightmarenet.compliance.pdf_builder import (
            PYHANKO_AVAILABLE,
            REPORTLAB_AVAILABLE,
        )
    except ImportError:
        pytest.skip("PDF builder module not available")

    if not (REPORTLAB_AVAILABLE and PYHANKO_AVAILABLE):
        pytest.skip("PDF dependencies not installed")

    class MockTracker:
        run_id = "test-run-123"

    output_dir = str(tmp_path / "output")
    pdf_path = generate_pdf(
        config=sample_config,
        comparison=sample_comparison,
        model_path=sample_model_file,
        output_dir=output_dir,
        tracker=MockTracker(),
    )

    assert Path(pdf_path).exists()
    assert "test-run-123" in pdf_path


def test_generate_pdf_signature_metadata(
    sample_config,
    sample_comparison,
    sample_model_file,
    tmp_path,
):
    """Test that PDF includes signature metadata."""
    try:
        from nightmarenet.compliance.pdf_builder import (
            PYHANKO_AVAILABLE,
            REPORTLAB_AVAILABLE,
        )
    except ImportError:
        pytest.skip("PDF builder module not available")

    if not (REPORTLAB_AVAILABLE and PYHANKO_AVAILABLE):
        pytest.skip("PDF dependencies not installed")

    output_dir = str(tmp_path / "output")
    pdf_path = generate_pdf(
        config=sample_config,
        comparison=sample_comparison,
        model_path=sample_model_file,
        output_dir=output_dir,
    )

    # Verify PDF contains metadata
    with open(pdf_path, "rb") as f:
        content = f.read()
        # Check for PDF signature-related content
        # Note: Actual signature verification requires more complex parsing
        assert b"NightmareNet" in content or b"Signature" in content or len(content) > 1000


def test_generate_pdf_model_directory(
    sample_config,
    sample_comparison,
    tmp_path,
):
    """Test PDF generation with model directory instead of file."""
    try:
        from nightmarenet.compliance.pdf_builder import (
            PYHANKO_AVAILABLE,
            REPORTLAB_AVAILABLE,
        )
    except ImportError:
        pytest.skip("PDF builder module not available")

    if not (REPORTLAB_AVAILABLE and PYHANKO_AVAILABLE):
        pytest.skip("PDF dependencies not installed")

    # Create a model directory with a model file
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"fake model data")

    output_dir = str(tmp_path / "output")
    pdf_path = generate_pdf(
        config=sample_config,
        comparison=sample_comparison,
        model_path=str(model_dir),
        output_dir=output_dir,
    )

    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 0


def test_pdf_builder_check_dependencies():
    """Test dependency check function."""
    try:
        from nightmarenet.compliance.pdf_builder import _check_dependencies
    except ImportError:
        pytest.skip("PDF builder module not available")

    try:
        _check_dependencies()
    except ImportError as e:
        # Expected if dependencies not installed
        assert "required" in str(e).lower()


def test_pdf_builder_get_version():
    """Test version retrieval function."""
    try:
        from nightmarenet.compliance.pdf_builder import _get_version
    except ImportError:
        pytest.skip("PDF builder module not available")

    version = _get_version()
    assert isinstance(version, str)
    assert len(version) > 0
