"""Tests for the demo corpus (data/demo/demo_corpus.csv)."""

from pathlib import Path

import pandas as pd
import pytest

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_corpus.csv"
REQUIRED_COLUMNS = [
    "id", "source", "url", "date", "title", "text", "lang",
    "lang_gold", "sentiment_gold",
]


@pytest.fixture(scope="module")
def corpus():
    return pd.read_csv(CORPUS_PATH)


def test_required_columns(corpus):
    """The corpus has every column the pipeline and the dashboard need."""
    for column in REQUIRED_COLUMNS:
        assert column in corpus.columns, f"missing column: {column}"


def test_ids_are_unique(corpus):
    """No two posts share an id."""
    assert corpus["id"].is_unique


def test_no_empty_texts(corpus):
    """Every post has some text."""
    assert corpus["text"].notna().all()
    assert (corpus["text"].str.strip() != "").all()


def test_lang_gold_values(corpus):
    """lang_gold only uses the five allowed labels."""
    assert set(corpus["lang_gold"]) <= {"bs", "hr", "sr", "en", "mixed"}
