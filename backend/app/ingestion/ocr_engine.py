
from pathlib import Path

import pymupdf
import pytesseract
from pytesseract import Output


SUPPORTED_LANGUAGES = {
    "english": "eng",
    "urdu": "urd",
    "sindhi": "snd"
}


def extract_text_with_ocr(
    file_path: str | Path,
    language: str = "english"
) -> list[dict]:
    """
    Extract text from scanned PDF pages using Tesseract OCR.

    Returns page-wise text, language, extraction method,
    and average OCR confidence.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError("Provided path is not a file.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose from {list(SUPPORTED_LANGUAGES)}"
        )

    lang_code = SUPPORTED_LANGUAGES[language]
    extracted_pages = []

    try:
        with pymupdf.open(str(pdf_path)) as document:

            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            if len(document) == 0:
                raise ValueError("PDF contains no pages.")

            for page_number, page in enumerate(document, start=1):

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = pixmap.pil_image()

                ocr_data = pytesseract.image_to_data(
                    image,
                    lang=lang_code,
                    output_type=Output.DICT
                )

                words = []
                confidences = []

                for word, confidence in zip(
                    ocr_data["text"],
                    ocr_data["conf"]
                ):
                    word = word.strip()
                    confidence = float(confidence)

                    if word and confidence >= 0:
                        words.append(word)
                        confidences.append(confidence)

                text = " ".join(words)

                average_confidence = (
                    round(sum(confidences) / len(confidences), 2)
                    if confidences
                    else 0.0
                )

                extracted_pages.append({
                    "page_number": page_number,
                    "text": text,
                    "language": language,
                    "extraction_method": "ocr",
                    "confidence": average_confidence
                })

        return extracted_pages

    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR is not available."
        ) from error
