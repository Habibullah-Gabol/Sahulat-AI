
import pytest

from backend.app.nlp.language_aware_chunker import chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("", language="english") == []


def test_single_sentence():
    result = chunk_text(
        "Sahulat AI helps users understand documents.",
        language="english"
    )

    assert len(result) == 1
    assert result[0]["chunk_index"] == 0
    assert result[0]["language"] == "english"


def test_multiple_sentences_are_grouped():
    text = (
        "First sentence. "
        "Second sentence. "
        "Third sentence."
    )

    result = chunk_text(
        text,
        language="english",
        max_chars=35,
        overlap_chars=5
    )

    assert len(result) >= 2
    assert all(item["text"] for item in result)


def test_urdu_sentence_boundaries():
    text = "یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔"

    result = chunk_text(
        text,
        language="urdu",
        max_chars=100,
        overlap_chars=20
    )

    assert len(result) == 1
    assert result[0]["language"] == "urdu"
    assert "یہ پہلا جملہ ہے۔" in result[0]["text"]
    assert "یہ دوسرا جملہ ہے۔" in result[0]["text"]


def test_sindhi_sentence_boundaries():
    text = "هي پهريون جملو آهي۔ هي ٻيو جملو آهي۔"

    result = chunk_text(
        text,
        language="sindhi",
        max_chars=100,
        overlap_chars=20
    )

    assert len(result) == 1
    assert result[0]["language"] == "sindhi"
    assert "هي پهريون جملو آهي۔" in result[0]["text"]
    assert "هي ٻيو جملو آهي۔" in result[0]["text"]


def test_chunk_metadata():
    result = chunk_text(
        "Document content.",
        language="english",
        source_id="document-001",
        page_number=3
    )

    assert result[0]["source_id"] == "document-001"
    assert result[0]["page_number"] == 3


def test_chunk_indices_are_sequential():
    text = "One sentence. Two sentence. Three sentence."

    result = chunk_text(
        text,
        language="english",
        max_chars=15,
        overlap_chars=3
    )

    assert [
        item["chunk_index"] for item in result
    ] == list(range(len(result)))


def test_long_sentence_is_split():
    text = " ".join(["document"] * 30)

    result = chunk_text(
        text,
        language="english",
        max_chars=50,
        overlap_chars=10
    )

    assert len(result) > 1
    assert all(item["char_count"] <= 50 for item in result)


def test_unsupported_language():
    with pytest.raises(ValueError):
        chunk_text("Some text.", language="french")


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text.",
            language="english",
            max_chars=0
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text.",
            language="english",
            max_chars=100,
            overlap_chars=100
        )


def test_invalid_page_number():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text.",
            language="english",
            page_number=0
        )


def test_non_string_text():
    with pytest.raises(TypeError):
        chunk_text(123, language="english")


def test_invalid_language_type():
    with pytest.raises(TypeError):
        chunk_text("Some text.", language=None)


def test_invalid_source_id_type():
    with pytest.raises(TypeError):
        chunk_text(
            "Some text.",
            language="english",
            source_id=123
        )
