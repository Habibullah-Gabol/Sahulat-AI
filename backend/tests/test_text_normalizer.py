
import pytest

from backend.app.nlp.text_normalizer import normalize_text


def test_english_text():
    assert normalize_text("  Hello   Sahulat AI!  ") == "Hello Sahulat AI!"


def test_urdu_yeh_normalization():
    assert normalize_text("علي", language="urdu") == "علی"


def test_urdu_kaf_normalization():
    assert normalize_text("كريم", language="urdu") == "کریم"


def test_sindhi_characters_preserved():
    text = "سنڌي ٻولي"
    assert normalize_text(text, language="sindhi") == text


def test_sindhi_arabic_yeh_preserved():
    text = "ي"
    assert normalize_text(text, language="sindhi") == text


def test_tatweel_removal():
    assert normalize_text("ســــلام") == "سلام"


def test_diacritics_preserved_by_default():
    text = "عَلِی"
    assert normalize_text(text) == text


def test_optional_diacritics_removal():
    assert normalize_text(
        "عَلِی",
        remove_diacritics=True
    ) == "علی"


def test_whitespace_normalization():
    assert normalize_text("سہولت\nAI\tپاکستان") == "سہولت AI پاکستان"


def test_punctuation_and_numbers_preserved():
    text = "Invoice #123: Rs. 5,000."
    assert normalize_text(text) == text


def test_empty_string():
    assert normalize_text("") == ""


def test_invalid_input():
    with pytest.raises(TypeError):
        normalize_text(123)


def test_invalid_diacritics_option():
    with pytest.raises(TypeError):
        normalize_text("سلام", remove_diacritics="yes")


def test_unsupported_language():
    with pytest.raises(ValueError):
        normalize_text("Hello", language="arabic")

def test_paragraph_breaks_are_preserved():
    text = "First paragraph.\n\nSecond paragraph."

    result = normalize_text(text)

    assert result == "First paragraph.\n\nSecond paragraph."
