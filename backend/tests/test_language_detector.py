
import pytest

from backend.app.ingestion.language_detector import detect_language


def test_detect_english():
    result = detect_language(
        "This is an official government document issued by the authorities."
    )

    assert result["language"] == "english"
    assert result["language_code"] == "en"
    assert result["needs_confirmation"] is False


def test_detect_urdu():
    result = detect_language(
        "یہ حکومت کی طرف سے جاری کردہ ایک سرکاری دستاویز ہے۔"
    )

    assert result["language"] == "urdu"
    assert result["language_code"] == "ur"


def test_detect_sindhi():
    result = detect_language(
        "هي حڪومت پاران جاري ڪيل هڪ سرڪاري دستاويز آهي۔"
    )

    assert result["language"] == "sindhi"
    assert result["language_code"] == "sd"


def test_low_confidence_requires_confirmation(monkeypatch):
    import backend.app.ingestion.language_detector as detector

    monkeypatch.setattr(
        detector,
        "detect",
        lambda **kwargs: {"lang": "sd", "score": 0.48}
    )

    result = detector.detect_language("سنڌي ٻولي")

    assert result["language"] == "sindhi"
    assert result["needs_confirmation"] is True


def test_empty_text():
    result = detect_language("")

    assert result["language"] == "undetermined"
    assert result["needs_confirmation"] is True


def test_invalid_input():
    with pytest.raises(TypeError):
        detect_language(None)
