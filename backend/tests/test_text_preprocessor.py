
import pytest

from backend.app.nlp.text_preprocessor import (
    split_paragraphs,
    split_sentences,
    preprocess_text
)


def test_paragraph_splitting():
    text = "First paragraph.\n\nSecond paragraph."
    result = split_paragraphs(text)

    assert result == [
        "First paragraph.",
        "Second paragraph."
    ]


def test_empty_paragraphs_removed():
    text = "\n\nFirst paragraph.\n\n\nSecond paragraph.\n\n"
    assert len(split_paragraphs(text)) == 2


def test_english_sentence_splitting():
    text = "Hello world. How are you? I am fine!"
    result = split_sentences(text)

    assert len(result) == 3


def test_urdu_sentence_splitting():
    text = "آپ کیسے ہیں؟ میں ٹھیک ہوں۔"
    result = split_sentences(text)

    assert len(result) == 2


def test_sindhi_sentence_splitting():
    text = "توهان ڪيئن آهيو؟ مان ٺيڪ آهيان۔"
    result = split_sentences(text)

    assert len(result) == 2


def test_legal_references_preserved():
    text = "The case was registered under Section 302. Further action is pending."

    result = split_sentences(text)

    assert len(result) == 2
    assert "Section 302" in result[0]


def test_dates_and_amounts_preserved():
    text = "The hearing is on 12/05/2026. The fee is Rs. 5,000."

    result = split_sentences(text)

    assert len(result) == 2
    assert "12/05/2026" in result[0]
    assert "Rs. 5,000" in result[1]


def test_percentage_preserved():
    text = "The tax rate is 17.5%. Payment is required."
    result = split_sentences(text)

    assert len(result) == 2
    assert "17.5%" in result[0]


def test_empty_sentence_input():
    assert split_sentences("   ") == []


def test_preprocess_output_structure():
    text = "Hello world.\n\nHow are you?"

    result = preprocess_text(text, language="english")

    assert result["language"] == "english"
    assert result["paragraph_count"] == 2
    assert result["sentence_count"] == 2
    assert len(result["paragraphs"]) == 2
    assert len(result["sentences"]) == 2


def test_preprocessing_uses_urdu_normalization():
    result = preprocess_text("علي", language="urdu")

    assert result["normalized_text"] == "علی"


def test_preprocessing_preserves_sindhi():
    text = "سنڌي ٻولي"

    result = preprocess_text(text, language="sindhi")

    assert result["normalized_text"] == text


def test_invalid_input():
    with pytest.raises(TypeError):
        preprocess_text(123)


def test_invalid_language():
    with pytest.raises(ValueError):
        preprocess_text("Hello", language="arabic")

