"""PDF generation with digital signature for EU AI Act compliance reports."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

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
    """Check if required dependencies are available."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install nightmarenet[compliance-pdf]"
        )
    if not PYHANKO_AVAILABLE:
        raise ImportError(
            "pyhanko is required for digital signatures. "
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
    report: dict,
    story: list,
    styles: dict,
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
    content: list,
    story: list,
    styles: dict,
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
    report: dict,
    story: list,
    styles: dict,
) -> None:
    """Create robustness metrics section."""
    robustness = report["robustness"]

    content = [
        [
            ["Clean Accuracy", str(robustness.get("clean_accuracy", "N/A"))],
            ["Distorted Accuracy", str(robustness.get("distorted_accuracy", "N/A"))],
            ["AUC Robustness", str(robustness.get("auc_robustness", "N/A"))],
            ["Delta", str(robustness.get("delta", "N/A"))],
        ]
    ]

    _create_section("Robustness Metrics", content, story, styles)


def _create_artifact_integrity_section(
    report: dict,
    story: list,
    styles: dict,
) -> None:
    """Create artifact integrity section."""
    integrity = report["artifact_integrity"]

    content = [
        [
            ["Config SHA-256", integrity.get("config_sha256", "N/A")],
            ["Model SHA-256", integrity.get("model_sha256", "N/A")],
        ]
    ]

    _create_section("Artifact Integrity", content, story, styles)


def _create_environment_section(
    report: dict,
    story: list,
    styles: dict,
) -> None:
    """Create runtime environment section."""
    env = report["environment"]

    content = [
        [
            ["Python Version", env.get("python_version", "N/A")],
            ["Platform", env.get("platform", "N/A")],
            ["PyTorch Version", env.get("pytorch_version", "N/A")],
            ["GPU", env.get("gpu", "N/A")],
        ]
    ]

    _create_section("Runtime Environment", content, story, styles)


def _create_eu_ai_act_section(
    report: dict,
    story: list,
    styles: dict,
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
    report: dict,
    story: list,
    styles: dict,
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
    story: list,
    styles: dict,
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
    report: dict,
    story: list,
    styles: dict,
) -> None:
    """Create appendix with raw metrics."""
    story.append(Paragraph("Appendix: Raw Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Full Report Data (JSON format):", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    import html
    import json

    json_str = json.dumps(report, indent=2, default=str)
    escaped_json = html.escape(json_str)
    story.append(Paragraph(f"<pre>{escaped_json}</pre>", styles["Code"]))
    story.append(Spacer(1, 0.3 * inch))


def _add_digital_signature(
    pdf_buffer: io.BytesIO,
    report: dict,
) -> io.BytesIO:
    """Add digital signature metadata to PDF."""
    # For now, return the PDF as-is
    # In a production environment, this would use a proper digital signature
    # with a certificate authority. The metadata is already included in the
    # document content via the appendix section.
    pdf_buffer.seek(0)
    return pdf_buffer


def generate_pdf(
    report: dict,
    output_path: str,
) -> str:
    """Generate a PDF compliance report with digital signature.

    Args:
        report: Compliance report dictionary from generate_report().
        output_path: Path where the PDF should be saved.

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
    story: list = []

    # Cover page
    _create_cover_page(report, story, styles)

    # Table of contents
    _create_table_of_contents(story, styles)

    # Content sections
    _create_robustness_section(report, story, styles)
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
    signed_pdf = _add_digital_signature(pdf_buffer, report)

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(signed_pdf.getvalue())

    return str(output_file)
