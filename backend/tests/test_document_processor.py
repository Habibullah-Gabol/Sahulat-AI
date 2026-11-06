
import pytest
import pymupdf

from reportlab.pdfgen import canvas
from pathlib import Path

from backend.app.ingestion.document_processor import process_document


def create_digital_pdf(file_path):
    pdf = canvas.Canvas(str(file_path))

    pdf.drawString(
        100, 750,
        "Sahulat AI digital document extraction test."
    )

    pdf.showPage()
    pdf.save()


def create_scanned_pdf(file_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 100),
        "SAHULAT AI SCANNED DOCUMENT TEST",
        fill="black"
    )

    image_path = file_path.parent / "scanned.png"
    image.save(image_path)

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_image(
        page.rect,
        filename=str(image_path)
    )

    pdf.save(str(file_path))
    pdf.close()


def create_mixed_pdf(file_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 100),
        "SAHULAT AI SCANNED PAGE",
        fill="black"
    )

    image_path = file_path.parent / "mixed.png"
    image.save(image_path)

    pdf = pymupdf.open()

    # Page 1: Digital text
    page1 = pdf.new_page()
    page1.insert_text(
        (50, 100),
        "This is a digital PDF page containing selectable text."
    )

    # Page 2: Scanned image
    page2 = pdf.new_page()
    page2.insert_image(
        page2.rect,
        filename=str(image_path)
    )

    pdf.save(str(file_path))
    pdf.close()


def test_process_digital_pdf(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    create_digital_pdf(pdf_path)

    result = process_document(pdf_path)

    assert result["total_pages"] == 1
    assert result["pages"][0]["extraction_method"] == "digital"
    assert "Sahulat AI" in result["pages"][0]["text"]


def test_process_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    create_scanned_pdf(pdf_path)

    result = process_document(pdf_path)

    assert result["total_pages"] == 1
    assert result["pages"][0]["extraction_method"] == "ocr"
    assert "SAHULAT" in result["pages"][0]["text"].upper()


def test_process_mixed_pdf(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    create_mixed_pdf(pdf_path)

    result = process_document(pdf_path)

    assert result["total_pages"] == 2

    assert (
        result["pages"][0]["extraction_method"] == "digital"
    )

    assert (
        result["pages"][1]["extraction_method"] == "ocr"
    )


def test_page_numbers_are_preserved(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    create_mixed_pdf(pdf_path)

    result = process_document(pdf_path)

    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][1]["page_number"] == 2


def test_unsupported_language(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    create_digital_pdf(pdf_path)

    with pytest.raises(ValueError, match="Unsupported OCR language"):
        process_document(pdf_path, ocr_language="french")


def test_invalid_document(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("Not a PDF")

    with pytest.raises(ValueError):
        process_document(pdf_path)
