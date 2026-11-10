
from ftlangdetect import detect


LANGUAGE_CODES = {
    "en": "english",
    "ur": "urdu",
    "sd": "sindhi"
}

CONFIDENCE_THRESHOLD = 0.70


def detect_language(text: str) -> dict:
    """
    Identify English, Urdu, and Sindhi using fastText LID-176.

    Low-confidence predictions are flagged for confirmation.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        return {
            "language": "undetermined",
            "confidence": 0.0,
            "method": "fasttext",
            "needs_confirmation": True
        }

    result = detect(text=text, low_memory=True)

    language_code = result["lang"]
    confidence = float(result["score"])

    language = LANGUAGE_CODES.get(
        language_code,
        "undetermined"
    )

    needs_confirmation = (
        language == "undetermined"
        or confidence < CONFIDENCE_THRESHOLD
    )

    return {
        "language": language,
        "language_code": language_code,
        "confidence": round(confidence, 4),
        "method": "fasttext",
        "needs_confirmation": needs_confirmation
    }
