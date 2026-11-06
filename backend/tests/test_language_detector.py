
import pytest

from backend.app.ingestion.language_detector import detect_language


def test_detect_english():
    result = detect_language(
        "This is an official government document."
    )

    assert result["language"] == "english"
    assert result["needs_confirmation"] is False


def test_detect_urdu():
    result = detect_language(
        "یہ ایک سرکاری دستاویز ہے۔"
    )

    assert result["language"] == "urdu"


def test_detect_sindhi():
    result = detect_language(
        "سنڌي ٻولي تمام خوبصورت آهي۔"
    )

    assert result["language"] == "sindhi"


def test_empty_text():
    result = detect_language("")

    assert result["language"] == "undetermined"
    assert result["needs_confirmation"] is True


def test_numbers_only():
    result = detect_language("123456789")

    assert result["language"] == "undetermined"


def test_ambiguous_arabic_script():
    result = detect_language("سلام")

    assert result["language"] == "undetermined"
    assert result["needs_confirmation"] is True


def test_invalid_input():
    with pytest.raises(TypeError):
        detect_language(None)
