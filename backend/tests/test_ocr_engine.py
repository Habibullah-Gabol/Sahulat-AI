
import pytest
import pymupdf

from backend.app.ingestion.ocr_engine import extract_text_with_ocr


def create_scanned_test_pdf(file_path):
    """Create a PDF containing an image of text."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 100), "SAHULAT AI OCR TEST 12345", fill="black")

    image_path = file_path.parent / "test_image.png"
    image.save(image_path)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, filename=str(image_path))
    document.save(str(file_path))
    document.close()


def test_english_ocr_extracts_text(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    create_scanned_test_pdf(pdf_path)

    result = extract_text_with_ocr(pdf_path, "english")

    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert result[0]["language"] == "english"
    assert "SAHULAT" in result[0]["text"].upper()


def test_urdu_ocr_runs(tmp_path):
    pdf_path = tmp_path / "urdu_test.pdf"
    create_scanned_test_pdf(pdf_path)

    result = extract_text_with_ocr(pdf_path, "urdu")

    assert len(result) == 1
    assert result[0]["language"] == "urdu"


def test_sindhi_ocr_runs(tmp_path):
    pdf_path = tmp_path / "sindhi_test.pdf"
    create_scanned_test_pdf(pdf_path)

    result = extract_text_with_ocr(pdf_path, "sindhi")

    assert len(result) == 1
    assert result[0]["language"] == "sindhi"


def test_unsupported_language(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    create_scanned_test_pdf(pdf_path)

    with pytest.raises(ValueError, match="Unsupported language"):
        extract_text_with_ocr(pdf_path, "french")


def test_missing_pdf():
    with pytest.raises(FileNotFoundError):
        extract_text_with_ocr("missing.pdf", "english")


def test_ocr_returns_confidence_metadata(tmp_path):
    pdf_path = tmp_path / "confidence.pdf"
    create_scanned_test_pdf(pdf_path)

    result = extract_text_with_ocr(pdf_path, "english")

    page = result[0]

    assert "confidence" in page
    assert isinstance(page["confidence"], float)
    assert 0 <= page["confidence"] <= 100


def test_ocr_returns_extraction_method(tmp_path):
    pdf_path = tmp_path / "method.pdf"
    create_scanned_test_pdf(pdf_path)

    result = extract_text_with_ocr(pdf_path, "english")

    assert result[0]["extraction_method"] == "ocr"
