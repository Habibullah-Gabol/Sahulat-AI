
from pathlib import Path

import pymupdf
import pytesseract
from pytesseract import Output

from backend.app.ingestion.validators import validate_pdf
from backend.app.ingestion.pdf_extractor import extract_text_from_pdf
from backend.app.ingestion.ocr_engine import SUPPORTED_LANGUAGES
from backend.app.ingestion.language_detector import detect_language


def process_document(
    file_path: str | Path,
    ocr_language: str | None = None,
    min_text_length: int = 20
) -> dict:

    if ocr_language is not None and ocr_language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported OCR language.")

    if min_text_length < 1:
        raise ValueError("min_text_length must be positive.")

    metadata = validate_pdf(file_path)
    digital_pages = extract_text_from_pdf(file_path)

    processed_pages = []

    with pymupdf.open(str(file_path)) as pdf:

        for index, page_data in enumerate(digital_pages):

            text = page_data["text"]
            is_scanned = len(text.strip()) < min_text_length

            if is_scanned:
                page = pdf[index]
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = pixmap.pil_image()

                # Detect language from page image using initial English OCR
                # unless the user specifies a language.
                if ocr_language:
                    selected_language = ocr_language
                    detection = None
                else:
                    preliminary_text = pytesseract.image_to_string(
                        image,
                        lang="eng+urd+snd"
                    )

                    detection = detect_language(preliminary_text)
                    selected_language = detection["language"]

                    if selected_language == "undetermined":
                        selected_language = "english"

                ocr_data = pytesseract.image_to_data(
                    image,
                    lang=SUPPORTED_LANGUAGES[selected_language],
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

                confidence = (
                    round(sum(confidences) / len(confidences), 2)
                    if confidences else 0.0
                )

                method = "ocr"

                if detection is None:
                    detection = {
                        "language": selected_language,
                        "confidence": None,
                        "method": "manual_selection",
                        "needs_confirmation": False
                    }

            else:
                detection = detect_language(text)
                selected_language = detection["language"]
                confidence = None
                method = "digital"

            processed_pages.append({
                "page_number": index + 1,
                "text": text,
                "extraction_method": method,
                "language": selected_language,
                "language_detection": detection,
                "confidence": confidence
            })

    return {
        "document": metadata,
        "pages": processed_pages,
        "total_pages": len(processed_pages)
    }
