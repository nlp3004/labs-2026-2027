"""Tests for text cleaning (radar/clean.py)."""

from radar.clean import clean_text


def test_removes_urls():
    """Links are removed."""
    assert "http" not in clean_text("Pogledaj ovo https://example.com/post/1 super")


def test_removes_mentions():
    """@mentions are removed."""
    assert "@" not in clean_text("@amra jesi li vidjela raspored?")


def test_keeps_diacritics():
    """č ć š ž đ are kept: they change meaning and help language ID."""
    assert clean_text("Šta ćemo raditi u četvrtak?") == "šta ćemo raditi u četvrtak?"


def test_keeps_emoji():
    """Emoji are kept: they carry sentiment."""
    assert "😡" in clean_text("Opet nema wifi 😡")


def test_collapses_repeated_punctuation():
    """"??!!" becomes a single "?"."""
    assert clean_text("Ozbiljno??!!") == "ozbiljno?"
