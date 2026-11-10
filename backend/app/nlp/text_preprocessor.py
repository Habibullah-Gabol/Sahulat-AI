
import re

from backend.app.nlp.text_normalizer import normalize_text


# Preserve common legal references such as Section 302 and Article 25-A.
LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|article|art\.?|rule|order|"
    r"act|clause|para\.?)\s+\d+[A-Za-z]?"
    r"(?:\([0-9A-Za-z]+\))*"
    r"(?:\s*,?\s*\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*)*",
    re.IGNORECASE
)

# Protect common date, time, and numeric patterns during sentence splitting.
PROTECTED_PATTERN = re.compile(
    r"""
    (?<!\w)
    (?:
        \d{1,4}[/-]\d{1,2}[/-]\d{1,4}
        |
        \d{1,2}:\d{2}(?:\s?[APap][Mm])?
        |
        Rs\.?\s?\d[\d,]*(?:\.\d{1,2})?
        |
        PKR\s?\d[\d,]*(?:\.\d{1,2})?
        |
        \d[\d,]*(?:\.\d{1,2})?%
    )
    (?!\w)
    """,
    re.IGNORECASE | re.VERBOSE
)


def split_paragraphs(text: str) -> list[str]:
    """Split document text into non-empty paragraphs."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    paragraphs = re.split(r"\n\s*\n", text.strip())

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_sentences(text: str) -> list[str]:
    """
    Split English, Urdu, and Sindhi text into sentences.

    Protects common numeric expressions and legal references
    from being split at internal punctuation.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        return []

    protected_items = []

    def protect(match):
        token = f"ZXQPROTECTED{len(protected_items)}QXZ"
        protected_items.append(match.group())
        return token

    # Protect legal references and common numeric expressions.
    text = LEGAL_REFERENCE_PATTERN.sub(protect, text)
    text = PROTECTED_PATTERN.sub(protect, text)

    # Split on common English and Urdu/Sindhi sentence endings.
    sentences = re.split(r"(?<=[.!?؟۔])\s+", text)

    restored_sentences = []

    for sentence in sentences:
        for index, original in enumerate(protected_items):
            sentence = sentence.replace(
                f"ZXQPROTECTED{index}QXZ",
                original
            )

        sentence = sentence.strip()

        if sentence:
            restored_sentences.append(sentence)

    return restored_sentences


def preprocess_text(
    text: str,
    language: str | None = None,
    remove_diacritics: bool = False
) -> dict:
    """
    Normalize text and organize it into paragraphs and sentences.
    """

    normalized_text = normalize_text(
        text,
        language=language,
        remove_diacritics=remove_diacritics
    )

    paragraphs = split_paragraphs(normalized_text)

    sentences = []

    for paragraph in paragraphs:
        sentences.extend(split_sentences(paragraph))

    return {
        "language": language or "undetermined",
        "normalized_text": normalized_text,
        "paragraphs": paragraphs,
        "sentences": sentences,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences)
    }
