"""PDF generation with digital signature for EU AI Act compliance reports."""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

PYHANKO_AVAILABLE = importlib.util.find_spec("pyhanko") is not None


def _check_dependencies() -> None:
    """Check if required dependencies are available for PDF generation."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install nightmarenet[compliance-pdf]"
        )


def _get_version() -> str:
    """Get NightmareNet version."""
    try:
        from importlib.metadata import version

        return version("nightmarenet")
    except Exception:
        return "0.2.0"


def _create_cover_page(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create cover page with report metadata."""
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=20,
        alignment=TA_CENTER,
    )

    story.append(Paragraph("EU AI Act Article 15", title_style))
    story.append(Paragraph("Compliance Report", title_style))
    story.append(Spacer(1, 0.5 * inch))

    story.append(Paragraph(f"Generated: {report['generated_at']}", subtitle_style))
    story.append(Paragraph(f"Schema Version: {report['schema_version']}", subtitle_style))
    story.append(Spacer(1, 0.5 * inch))

    model_info = [
        ["Model Name", report["model"].get("name") or "N/A"],
        ["Model Type", report["model"].get("type") or "N/A"],
        ["Dataset", report["dataset"].get("name") or "N/A"],
        ["Model Name", report["model"].get("name", "N/A")],
        ["Model Type", report["model"].get("type", "N/A")],
        ["Dataset", report["dataset"].get("name", "N/A")],
    ]

    table = Table(model_info, colWidths=[2 * inch, 4 * inch])
    table.setStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
        ]
    )
    story.append(table)
    story.append(PageBreak())


def _create_section(
    title: str,
    content: list[Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create a section with title and content."""
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    for item in content:
        if item is None:
            continue
        if isinstance(item, str):
            story.append(Paragraph(item, styles["Normal"]))
        elif isinstance(item, list):
            # Convert None values to "N/A" strings and filter out None rows
            normalized_item = []
            for row in item:
                if row is None:
                    continue
                normalized_row = [str(cell) if cell is not None else "N/A" for cell in row]
                normalized_item.append(normalized_row)

            if normalized_item:
                table = Table(normalized_item, colWidths=[2 * inch, 4 * inch])
                table.setStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
                story.append(table)

    story.append(Spacer(1, 0.3 * inch))


def _create_robustness_section(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create robustness metrics section."""
    robustness = report["robustness"]

    content = [
        ["Clean Accuracy", str(robustness.get("clean_accuracy", "N/A"))],
        ["Distorted Accuracy", str(robustness.get("distorted_accuracy", "N/A"))],
        ["AUC Robustness", str(robustness.get("auc_robustness", "N/A"))],
        ["Delta", str(robustness.get("delta", "N/A"))],
        [
            ["Clean Accuracy", str(robustness.get("clean_accuracy", "N/A"))],
            ["Distorted Accuracy", str(robustness.get("distorted_accuracy", "N/A"))],
            ["AUC Robustness", str(robustness.get("auc_robustness", "N/A"))],
            ["Delta", str(robustness.get("delta", "N/A"))],
        ]
    ]

    _create_section("Robustness Metrics", content, story, styles)


def _create_artifact_integrity_section(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create artifact integrity section."""
    integrity = report["artifact_integrity"]

    content = [
        ["Config SHA-256", integrity.get("config_sha256", "N/A")],
        ["Model SHA-256", integrity.get("model_sha256", "N/A")],
        [
            ["Config SHA-256", integrity.get("config_sha256", "N/A")],
            ["Model SHA-256", integrity.get("model_sha256", "N/A")],
        ]
    ]

    _create_section("Artifact Integrity", content, story, styles)


def _create_environment_section(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create runtime environment section."""
    env = report["environment"]

    content = [
        ["Python Version", env.get("python_version", "N/A")],
        ["Platform", env.get("platform", "N/A")],
        ["PyTorch Version", env.get("pytorch_version", "N/A")],
        ["GPU", env.get("gpu", "N/A")],
        [
            ["Python Version", env.get("python_version", "N/A")],
            ["Platform", env.get("platform", "N/A")],
            ["PyTorch Version", env.get("pytorch_version", "N/A")],
            ["GPU", env.get("gpu", "N/A")],
        ]
    ]

    _create_section("Runtime Environment", content, story, styles)


def _create_eu_ai_act_section(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create EU AI Act mapping section."""
    eu_mapping = report["eu_ai_act"]

    story.append(Paragraph("EU AI Act Article 15 Mapping", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    for key, value in eu_mapping["requirements"].items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Spacer(1, 0.3 * inch))


def _create_nist_section(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create NIST AI RMF mapping section."""
    nist_mapping = report["nist_ai_rmf"]

    story.append(Paragraph("NIST AI RMF Mapping", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    for section, items in nist_mapping.items():
        story.append(Paragraph(section, styles["Heading3"]))
        for item in items:
            story.append(Paragraph(f"- {item}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Spacer(1, 0.3 * inch))


def _create_table_of_contents(
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create a manual table of contents."""
    story.append(Paragraph("Table of Contents", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    toc_items = [
        ["1. Robustness Metrics", "3"],
        ["2. Artifact Integrity", "4"],
        ["3. Runtime Environment", "5"],
        ["4. EU AI Act Article 15 Mapping", "6"],
        ["5. NIST AI RMF Mapping", "7"],
        ["6. Appendix: Raw Metrics", "8"],
    ]

    toc_table = Table(toc_items, colWidths=[3 * inch, 1 * inch])
    toc_table.setStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]
    )
    story.append(toc_table)
    story.append(PageBreak())


def _create_appendix(
    report: dict[str, Any],
    story: list[Any],
    styles: dict[str, Any],
) -> None:
    """Create appendix with raw metrics."""
    story.append(Paragraph("Appendix: Raw Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Full Report Data (JSON format):", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    import json

    json_str = json.dumps(report, indent=2, default=str)
    story.append(Paragraph(f"<pre>{json_str}</pre>", styles["Code"]))
    import html
    import json

    json_str = json.dumps(report, indent=2, default=str)
    escaped_json = html.escape(json_str)
    story.append(Paragraph(f"<pre>{escaped_json}</pre>", styles["Code"]))
    story.append(Spacer(1, 0.3 * inch))


def _generate_ephemeral_cert() -> tuple[str, str]:
    """Generate an ephemeral self-signed certificate and private key in temporary files.

    Returns:
        Tuple of (cert_file_path, key_file_path).
    """
    from datetime import timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "NightmareNet Compliance Engine"),
        ]
    )
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .sign(key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    cert_file = tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False)
    key_file = tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False)
    os.chmod(key_file.name, 0o600)

    try:
        cert_file.write(cert_pem)
        cert_file.flush()
        key_file.write(key_pem)
        key_file.flush()
    finally:
        cert_file.close()
        key_file.close()

    return cert_file.name, key_file.name


def _add_digital_signature(
    pdf_buffer: io.BytesIO,
    report: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> io.BytesIO:
    """Add a cryptographic digital signature to the PDF using pyHanko.

    Supports custom certificate configured via compliance.signing_cert_path
    or generates an ephemeral self-signed certificate. Falls back gracefully
    to returning the unsigned PDF buffer on any error.
    """
    pdf_buffer.seek(0)

    if not PYHANKO_AVAILABLE:
        logger.warning("pyHanko unavailable — returning unsigned PDF report.")
        return pdf_buffer

    cert_path = None
    if config:
        tracking_cfg = config.get("tracking", {})
        compliance_cfg = tracking_cfg.get("compliance", {})
        if isinstance(compliance_cfg, dict):
            cert_path = compliance_cfg.get("signing_cert_path")

    tmp_cert_path = None
    tmp_key_path = None

    try:
        from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
        from pyhanko.sign import signers

        if cert_path and Path(cert_path).exists():
            signer = signers.SimpleSigner.load(key_file=cert_path, cert_file=cert_path)
        else:
            if cert_path:
                logger.warning(
                    "Configured signing_cert_path '%s' not found. "
                    "Falling back to ephemeral self-signed certificate.",
                    cert_path,
                )
            tmp_cert_path, tmp_key_path = _generate_ephemeral_cert()
            signer = signers.SimpleSigner.load(key_file=tmp_key_path, cert_file=tmp_cert_path)

        writer = IncrementalPdfFileWriter(pdf_buffer)
        sig_meta = signers.PdfSignatureMetadata(
            field_name="Signature1",
            reason="EU AI Act Article 15 Compliance Report",
            location="NightmareNet Compliance Engine",
        )

        signed_buffer = io.BytesIO()
        signers.sign_pdf(writer, sig_meta, signer=signer, output=signed_buffer)
        signed_buffer.seek(0)
        return signed_buffer

    except Exception as e:
        logger.warning("PDF digital signature failed: %s. Returning unsigned PDF.", e)
        pdf_buffer.seek(0)
        return pdf_buffer

    finally:
        for path in (tmp_cert_path, tmp_key_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


def generate_pdf(
    report: dict[str, Any],
    output_path: str,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a PDF compliance report with digital signature.

    Args:
        report: Compliance report dictionary from generate_report().
        output_path: Path where the PDF should be saved.
        config: Optional configuration dictionary.

    Returns:
        Path to the generated PDF file.

    Raises:
        ImportError: If required dependencies are not installed.
    """
    _check_dependencies()

    # Create PDF buffer
    pdf_buffer = io.BytesIO()

    # Setup document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Setup styles
    styles = getSampleStyleSheet()
    if "Code" not in styles:
        styles.add(
            ParagraphStyle(
                name="Code",
                fontName="Courier",
                fontSize=8,
                leading=10,
            )
        )
    else:
        styles["Code"].fontName = "Courier"
        styles["Code"].fontSize = 8
        styles["Code"].leading = 10

    # Build story
    story: list[Any] = []

    # Cover page
    _create_cover_page(report, story, styles)

    # Table of contents
    _create_table_of_contents(story, styles)

    # Content sections
    _create_robustness_section(report, story, styles)
    _create_artifact_integrity_section(report, story, styles)
    _create_environment_section(report, story, styles)
    _create_eu_ai_act_section(report, story, styles)
    _create_nist_section(report, story, styles)
    story.append(PageBreak())
    _create_artifact_integrity_section(report, story, styles)
    story.append(PageBreak())
    _create_environment_section(report, story, styles)
    story.append(PageBreak())
    _create_eu_ai_act_section(report, story, styles)
    story.append(PageBreak())
    _create_nist_section(report, story, styles)
    story.append(PageBreak())
    _create_appendix(report, story, styles)

    # Build PDF
    doc.build(story)

    # Add digital signature
    signed_pdf = _add_digital_signature(pdf_buffer, report, config=config)

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(signed_pdf.getvalue())

    return str(output_file)
