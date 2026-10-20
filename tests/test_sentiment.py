"""Unit tests — runnable WITHOUT downloading the transformer model.

Covers: empty/whitespace/max-length validation, result structure, label
mapping, probability range + sum, error handling, unicode/emoji acceptance.

Integration tests (real model) are marked and skipped gracefully offline:
    python -m pytest -q -m "not integration"
"""

import math

import pytest

from sentiment_model import (
    MAX_CHARS,
    build_result,
    keyword_fallback_sentiment,
    normalize_id2label,
    normalize_label,
    softmax,
    validate_text,
)


# --- validation -----------------------------------------------------------

def test_empty_string_rejected():
    _, err = validate_text("")
    assert err


def test_whitespace_only_rejected():
    _, err = validate_text("   \n\t  ")
    assert err


def test_valid_text_accepted_and_stripped():
    cleaned, err = validate_text("  hello ❤️  ")
    assert err is None
    assert cleaned == "hello ❤️"


def test_too_long_rejected():
    _, err = validate_text("x" * (MAX_CHARS + 1))
    assert err and "too long" in err.lower()


def test_non_string_rejected():
    _, err = validate_text(None)
    assert err


def test_unicode_emoji_accepted():
    cleaned, err = validate_text("😂❤️🔥 Great job!!! #win @team")
    assert err is None
    assert "😂" in cleaned


def test_hashtags_mentions_punctuation_accepted():
    cleaned, err = validate_text("Not bad at all... @user #ok?!")
    assert err is None
    assert cleaned.startswith("Not bad")


# --- label mapping --------------------------------------------------------

def test_normalize_direct_labels():
    assert normalize_label("positive") == "POSITIVE"
    assert normalize_label("NEGATIVE") == "NEGATIVE"
    assert normalize_label("Neutral") == "NEUTRAL"


def test_normalize_generic_label_ids():
    assert normalize_label("LABEL_0", 0) == "NEGATIVE"
    assert normalize_label("LABEL_1", 1) == "NEUTRAL"
    assert normalize_label("LABEL_2", 2) == "POSITIVE"


def test_normalize_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_label("mystery")


def test_id2label_mapping():
    mapped = normalize_id2label({0: "negative", 1: "neutral", 2: "positive"})
    assert mapped == {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def test_id2label_empty_raises():
    with pytest.raises(ValueError):
        normalize_id2label({})


# --- result structure / probabilities -------------------------------------

def test_build_result_structure():
    r = build_result("POSITIVE", positive=0.94, neutral=0.04, negative=0.02)
    assert r["sentiment"] == "POSITIVE"
    assert r["confidence"] == pytest.approx(0.94)
    assert set(r["scores"]) == {"positive", "neutral", "negative"}


def test_probabilities_in_range_and_sum_to_one():
    r = build_result("NEUTRAL", positive=0.25, neutral=0.50, negative=0.25)
    for v in r["scores"].values():
        assert 0.0 <= v <= 1.0
    assert math.isclose(sum(r["scores"].values()), 1.0, abs_tol=0.01)


def test_build_result_bad_sum_raises():
    with pytest.raises(ValueError):
        build_result("POSITIVE", positive=0.5, neutral=0.5, negative=0.5)


def test_build_result_bad_sentiment_raises():
    with pytest.raises(ValueError):
        build_result("HAPPY", positive=0.5, neutral=0.3, negative=0.2)


def test_softmax_sums_to_one():
    probs = softmax([2.0, 1.0, 0.1])
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
    assert probs[0] > probs[1] > probs[2]


# --- error handling / fallback --------------------------------------------

def test_analyze_empty_raises_without_model():
    from sentiment_model import analyze_sentiment

    with pytest.raises(ValueError):
        analyze_sentiment("   ")


def test_fallback_is_labeled_and_structured():
    r = keyword_fallback_sentiment("I love this! ❤️")
    assert r["sentiment"] == "POSITIVE"
    assert r["fallback"] is True
    assert r["model"] == "keyword-fallback"
    assert math.isclose(sum(r["scores"].values()), 1.0, abs_tol=0.01)


def test_fallback_negative_and_neutral():
    assert keyword_fallback_sentiment("I hate this 😡")["sentiment"] == "NEGATIVE"
    assert keyword_fallback_sentiment("The package arrived today.")["sentiment"] == "NEUTRAL"


def test_fallback_rejects_empty():
    with pytest.raises(ValueError):
        keyword_fallback_sentiment("")


# --- integration (real model; skipped gracefully offline) ------------------

@pytest.mark.integration
def test_real_model_positive():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from sentiment_model import analyze_sentiment

    try:
        r = analyze_sentiment("I love this product.")
    except RuntimeError as exc:
        pytest.skip(f"Model unavailable: {exc}")
    assert r["sentiment"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    assert math.isclose(sum(r["scores"].values()), 1.0, abs_tol=0.01)


@pytest.mark.integration
def test_real_model_demo_set():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from sentiment_model import analyze_sentiment

    demo = [
        "I love this product.",
        "I hate this product.",
        "The package arrived today.",
        "😍❤️🔥",
    ]
    for text in demo:
        try:
            r = analyze_sentiment(text)
        except RuntimeError as exc:
            pytest.skip(f"Model unavailable: {exc}")
        print(f"\n{text!r} -> {r['sentiment']} ({r['confidence']:.2%})")
        assert r["sentiment"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
