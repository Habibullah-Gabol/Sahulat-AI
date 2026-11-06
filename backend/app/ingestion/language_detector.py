
import re


LANGUAGE_PATTERNS = {
    "urdu": set("ٹڈڑںھۂے"),
    "sindhi": set("ٻڄڏڊڙڳڪڱڻ"),
}


def detect_language(text: str) -> dict:
    """
    Conservatively detect English, Urdu, or Sindhi.

    Returns an undetermined result when the available
    text does not provide enough evidence.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        return {
            "language": "undetermined",
            "confidence": 0.0,
            "method": "character_analysis",
            "needs_confirmation": True
        }

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return {
            "language": "undetermined",
            "confidence": 0.0,
            "method": "character_analysis",
            "needs_confirmation": True
        }

    latin_count = sum(
        "LATIN" in __import__("unicodedata").name(char, "")
        for char in letters
    )

    arabic_count = sum(
        "ARABIC" in __import__("unicodedata").name(char, "")
        for char in letters
    )

    total_letters = len(letters)

    latin_ratio = latin_count / total_letters
    arabic_ratio = arabic_count / total_letters

    # English detection
    if latin_ratio >= 0.70:
        return {
            "language": "english",
            "confidence": round(latin_ratio, 2),
            "method": "script_analysis",
            "needs_confirmation": False
        }

    # Urdu and Sindhi character signals
    urdu_score = sum(
        char in LANGUAGE_PATTERNS["urdu"]
        for char in text
    )

    sindhi_score = sum(
        char in LANGUAGE_PATTERNS["sindhi"]
        for char in text
    )

    # Sindhi-specific character evidence
    if sindhi_score > 0:
        return {
            "language": "sindhi",
            "confidence": min(
                0.95,
                round(0.70 + sindhi_score * 0.05, 2)
            ),
            "method": "character_analysis",
            "needs_confirmation": False
        }

    # Urdu-specific character evidence
    if urdu_score > 0:
        return {
            "language": "urdu",
            "confidence": min(
                0.95,
                round(0.70 + urdu_score * 0.05, 2)
            ),
            "method": "character_analysis",
            "needs_confirmation": False
        }

    # Arabic-derived script without distinguishing evidence
    if arabic_ratio > 0.50:
        return {
            "language": "undetermined",
            "confidence": 0.0,
            "method": "script_analysis",
            "needs_confirmation": True
        }

    return {
        "language": "undetermined",
        "confidence": 0.0,
        "method": "character_analysis",
        "needs_confirmation": True
    }
