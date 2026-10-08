"""Tests for sentiment analysis (radar/sentiment.py)."""

import pytest

from radar.sentiment import MODEL_DIR, sentiment_vader, sentiment_xlmr


@pytest.mark.parametrize("text, expected", [
    ("I love this course, it is great!", "positive"),
    ("This is terrible, I hate it.", "negative"),
    ("The lecture starts at 10.", "neutral"),
])
def test_vader_clear_english(text, expected):
    """VADER labels clear English sentences correctly."""
    label, _ = sentiment_vader(text)
    assert label == expected


@pytest.mark.slow
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="XLM-R not downloaded (scripts/download_models.py)")
def test_xlmr_clear_examples():
    """XLM-R labels a clear positive and a clear negative post correctly."""
    labels, _ = sentiment_xlmr(["Predavanje je bilo odlično, hvala!", "Ovo je katastrofa, užas."])
    assert labels == ["positive", "negative"]
