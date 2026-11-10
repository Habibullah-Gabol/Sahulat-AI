
import re
import unicodedata

URDU_CHARACTER_MAP = {
    "\u064A": "\u06CC",
    "\u0649": "\u06CC",
    "\u0643": "\u06A9",
}

DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")


def normalize_text(
    text: str,
    language: str | None = None,
    remove_diacritics: bool = False
) -> str:
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if language not in (None, "english", "urdu", "sindhi"):
        raise ValueError("Unsupported language.")

    if not isinstance(remove_diacritics, bool):
        raise TypeError("remove_diacritics must be a boolean.")

    text = unicodedata.normalize("NFC", text)

    if language == "urdu":
        text = "".join(
            URDU_CHARACTER_MAP.get(char, char)
            for char in text
        )

    text = text.replace("\u0640", "")

    if remove_diacritics:
        text = DIACRITICS_PATTERN.sub("", text)

    # Preserve paragraph boundaries and clean other whitespace.
    paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs = [
        re.sub(r"[^\S\n]+", " ", paragraph)
        .replace("\n", " ")
        .strip()
        for paragraph in paragraphs
    ]

    return "\n\n".join(
        paragraph
        for paragraph in cleaned_paragraphs
        if paragraph
    )
