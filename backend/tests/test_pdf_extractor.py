
import pytest
from pathlib import Path
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from backend.app.ingestion.pdf_extractor import extract_text_from_pdf


def create_test_pdf(file_path: Path) -> None:
    """Create a blank PDF for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)


def test_missing_pdf():
    """Verify that a missing PDF raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("missing.pdf")


def test_invalid_file_format(tmp_path):
    """Verify that non-PDF files are rejected."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Sample content")

    with pytest.raises(ValueError, match="Only PDF"):
        extract_text_from_pdf(text_file)


def test_pdf_page_count(tmp_path):
    """Verify that the extractor returns one result per PDF page."""
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    result = extract_text_from_pdf(pdf_path)

    assert len(result) == 1
    assert result[0]["page_number"] == 1


def test_page_structure(tmp_path):
    """Verify that extracted results contain the expected fields."""
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    result = extract_text_from_pdf(pdf_path)

    assert "page_number" in result[0]
    assert "text" in result[0]
    assert isinstance(result[0]["text"], str)


def test_actual_text_extraction(tmp_path):
    """Verify that actual text is extracted from a PDF."""
    pdf_path = tmp_path / "text_sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(100, 750, "Welcome to Sahulat AI")
    pdf.save()

    result = extract_text_from_pdf(pdf_path)

    assert len(result) == 1
    assert "Welcome to Sahulat AI" in result[0]["text"]

def test_multi_page_pdf_extraction(tmp_path):
    """Verify extraction and page numbering across multiple pages."""
    pdf_path = tmp_path / "multi_page.pdf"

    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(100, 750, "First page content")
    pdf.showPage()

    pdf.drawString(100, 750, "Second page content")
    pdf.showPage()

    pdf.drawString(100, 750, "Third page content")
    pdf.save()

    result = extract_text_from_pdf(pdf_path)

    assert len(result) == 3
    assert [page["page_number"] for page in result] == [1, 2, 3]

    assert "First page content" in result[0]["text"]
    assert "Second page content" in result[1]["text"]
    assert "Third page content" in result[2]["text"]
