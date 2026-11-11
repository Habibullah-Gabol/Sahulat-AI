
import re


SUPPORTED_LANGUAGES = {"english", "urdu", "sindhi"}

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?؟۔])\s+")


def _validate_settings(max_chars: int, overlap_chars: int) -> None:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise TypeError("max_chars must be an integer.")

    if not isinstance(overlap_chars, int) or isinstance(overlap_chars, bool):
        raise TypeError("overlap_chars must be an integer.")

    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")


def _split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph using common English, Urdu and Sindhi endings."""
    return [
        sentence.strip()
        for sentence in SENTENCE_ENDINGS.split(paragraph)
        if sentence.strip()
    ]


def _split_long_sentence(
    sentence: str,
    max_chars: int,
    overlap_chars: int
) -> list[str]:
    """Split long sentences at word boundaries with character overlap."""

    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    pieces = []
    current_words = []
    current_length = 0

    for word in words:
        added_length = len(word) if not current_words else len(word) + 1

        if current_words and current_length + added_length > max_chars:
            pieces.append(" ".join(current_words))

            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):
                added = len(previous_word) + (1 if overlap_words else 0)

                if overlap_length + added > overlap_chars:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += added

            current_words = overlap_words
            current_length = len(" ".join(current_words))

        current_words.append(word)
        current_length = len(" ".join(current_words))

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def chunk_text(
    text: str,
    language: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
    source_id: str | None = None,
    page_number: int | None = None
) -> list[dict]:
    """
    Split text into overlapping, sentence-aware chunks.

    Args:
        text: Document text.
        language: english, urdu or sindhi.
        max_chars: Maximum preferred chunk length in characters.
        overlap_chars: Approximate overlap between split sections.
        source_id: Optional document identifier.
        page_number: Optional source page number.

    Returns:
        A list of chunk dictionaries with text and source metadata.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(language, str):
        raise TypeError("language must be a string.")

    language = language.strip().lower()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported language.")

    _validate_settings(max_chars, overlap_chars)

    if source_id is not None and not isinstance(source_id, str):
        raise TypeError("source_id must be a string or None.")

    if page_number is not None:
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise TypeError("page_number must be an integer or None.")
        if page_number < 1:
            raise ValueError("page_number must be positive.")

    text = text.strip()

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    units = []

    for paragraph in paragraphs:
        for sentence in _split_sentences(paragraph):
            units.extend(
                _split_long_sentence(
                    sentence,
                    max_chars,
                    overlap_chars
                )
            )

    chunks = []
    current_parts = []
    current_length = 0

    for unit in units:
        separator = " " if current_parts else ""
        proposed_length = current_length + len(separator) + len(unit)

        if current_parts and proposed_length > max_chars:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_length = 0

        if len(unit) > max_chars:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_length = 0

            chunks.append(unit)
            continue

        if current_parts:
            current_length += 1

        current_parts.append(unit)
        current_length += len(unit)

    if current_parts:
        chunks.append(" ".join(current_parts))

    results = []

    for index, chunk in enumerate(chunks):
        results.append({
            "chunk_index": index,
            "text": chunk,
            "language": language,
            "char_count": len(chunk),
            "source_id": source_id,
            "page_number": page_number
        })

    return results
