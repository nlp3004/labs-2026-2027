"""Stage 2 - Text cleaning (normalisation).

NLP step: turning messy raw text into a more uniform version for analysis.
Two posts that mean the same thing ("WIFI!!!" and "wifi!") should look the same
to our tools. Cleaning removes noise, but every removal also throws away some
information, so we decide carefully what to remove and what to keep.

We never overwrite the original text. Cleaning produces a COPY for analysis,
so we can always go back and see what the user actually wrote.
"""

import re

# Patterns are compiled once here, not on every call.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
# Two or more punctuation marks in a row, e.g. "??!!" or "!!!" or "...".
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,])[!?.,]+")
WHITESPACE_PATTERN = re.compile(r"\s+")

DIACRITIC_LETTERS = set("čćšžđČĆŠŽĐ")
# How people type č ć š ž đ on a keyboard without them.
DIACRITICS_TO_PLAIN = str.maketrans({
    "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
    "Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "Dj",
})


def clean_text(text):
    """NLP step: text normalisation - make a cleaned, lowercase copy of a post.

    Why it matters: tools see "Wifi", "WIFI" and "wifi" as three different
    words unless we normalise them. URLs and @mentions are not language, so they
    only confuse language and sentiment tools.

    What we deliberately KEEP:
      - Emoji: they carry sentiment. "opet nema wifi 😡" and "opet nema wifi 😂"
        mean different things, and sentiment tools like VADER read emoji.
      - Diacritics (č ć š ž đ): they change meaning ("kuća" = house,
        "kuca" = knocks) and they are strong clues for language identification.
        Removing them would make our data look like the hardest posts.
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    # "??!!" -> "?": the repetition adds emphasis, not new words.
    text = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def has_diacritics(text):
    """NLP step: a simple text feature - does the post contain č ć š ž đ?

    Why it matters: many people type BCS without diacritics ("sta" instead of
    "šta"). We use this to check whether our tools do worse on those posts.
    """
    return any(char in DIACRITIC_LETTERS for char in text)


def strip_diacritics(text):
    """NLP step: a normalisation helper - write č ć š ž đ as c c s z dj.

    Why it matters: we do NOT apply this to our cleaned text (see clean_text).
    It is only used to compare words fairly: "šta" and "sta" are the same
    word, typed with and without diacritics.
    """
    return text.translate(DIACRITICS_TO_PLAIN)
