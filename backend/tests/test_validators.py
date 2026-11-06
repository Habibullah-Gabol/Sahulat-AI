
import pytest
from pathlib import Path
from reportlab.pdfgen import canvas

from backend.app.ingestion.validators import validate_pdf


def create_test_pdf(file_path: Path):
    """Create a small valid PDF for testing."""
    pdf = canvas.Canvas(str(file_path))
    pdf.drawString(100, 750, "Sahulat AI Test Document")
    pdf.save()


def test_validate_valid_pdf(tmp_path):
    pdf_path = tmp_path / "valid.pdf"
    create_test_pdf(pdf_path)

    result = validate_pdf(pdf_path)

    assert result["valid"] is True
    assert result["filename"] == "valid.pdf"
    assert result["page_count"] == 1
    assert result["file_type"] == "application/pdf"


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_pdf("missing.pdf")


def test_invalid_extension(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(ValueError, match="Only PDF"):
        validate_pdf(file_path)


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.pdf"
    file_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        validate_pdf(file_path)


def test_corrupted_pdf(tmp_path):
    file_path = tmp_path / "corrupted.pdf"
    file_path.write_bytes(b"This is not a valid PDF")

    with pytest.raises(ValueError, match="Invalid PDF"):
        validate_pdf(file_path)


def test_file_exceeds_size_limit(tmp_path):
    from backend.app.ingestion.validators import MAX_FILE_SIZE

    file_path = tmp_path / "large.pdf"
    file_path.write_bytes(b"A" * (MAX_FILE_SIZE + 1))

    with pytest.raises(ValueError, match="maximum size"):
        validate_pdf(file_path)
