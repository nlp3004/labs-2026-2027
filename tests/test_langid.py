"""Tests for our own language detector (radar/langid.py)."""

import pytest

from radar.langid import detect_ours

# Clear examples: each one has words that only one language uses.
EXAMPLES = [
    ("Tko zna kad je kolokvij ovaj tjedan?", "hr"),
    ("Šta ima novo, ko ide u menzu ove sedmice?", "bs"),
    ("sta ima, ko ide sutra u menzu ove sedmice?", "bs"),  # typed without diacritics
    ("Da li neko zna gde je biblioteka? Treba da vratim knjigu.", "sr"),
    ("Does anyone know when the library opens tomorrow?", "en"),
    ("exam je bio so hard, ne mogu vise", "mixed"),
]


@pytest.mark.parametrize("text, expected", EXAMPLES)
def test_detect_ours_clear_examples(text, expected):
    """detect_ours gets clear, textbook examples right."""
    assert detect_ours(text) == expected
