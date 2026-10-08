"""Stage 3 - Language identification.

NLP step: deciding which language a text is written in. Almost every later step
depends on it: a sentiment model for English cannot read Bosnian, so we must know
the language before we pick a tool.

Bosnian, Croatian and Serbian (BCS) are very close. They share most of their
words, so a short post can be valid in all three. That makes BCS a great test
for language identification tools. We compare three detectors:

  a) detect_naive  - `langdetect`, a popular off-the-shelf library
  b) detect_lingua - `lingua`, limited to the 4 languages we expect
  c) detect_ours   - our own rules: marker words that differ between bs/hr/sr
"""

from langdetect import DetectorFactory, LangDetectException, detect
from lingua import Language, LanguageDetectorBuilder

from radar.clean import strip_diacritics

# =============================================================================
# Rules for detect_ours. Students: edit these and re-run the pipeline!
# =============================================================================

# Marker words that point to one language. Some words are typical for two
# languages, so they appear in two lists (e.g. "šta" is Bosnian AND Serbian).
# Write them WITH diacritics. Posts typed without diacritics still match,
# because we compare both sides without diacritics.
# Phrases with a space ("da li") are allowed.
#
# HONEST TUNING: look for new marker words only in the "dev" posts
# (split == "dev"). The "test" posts are for the final score. If you copy
# words from test posts into these lists, your test score stops meaning
# anything: you would be grading yourself on answers you have already seen.
# The words below are general differences between the languages, the kind
# you would find in a grammar book, not words copied from particular posts.
MARKER_WORDS = {  # LAB 1: change me
    "bs": [
        # ijekavian forms (also Croatian): mlijeko, not mleko
        "mlijeko", "lijepo", "vrijeme", "prije", "poslije", "gdje", "dvije",
        "mjesto", "mjesta",
        # ko / šta (also Serbian), not tko / što
        "ko", "neko", "niko", "šta",
        # typically Bosnian words
        "hljeb", "sedmica", "sedmice", "sedmicu", "kahva", "jok", "haman",
        "merak", "bona", "ba", "jel", "hemija",
    ],
    "hr": [
        # ijekavian forms (also Bosnian)
        "mlijeko", "lijepo", "vrijeme", "prije", "poslije", "gdje", "dvije",
        "mjesto", "mjesta",
        # tko / što, not ko / šta
        "tko", "netko", "nitko", "što",
        # "je li" / "zna li" questions, not "da li"
        "je li", "zna li", "trebam li", "hoću li",
        # typically Croatian words
        "kruh", "tjedan", "tjedna", "tjednu", "kava", "kavu", "juha", "juhu",
        "knjižnica", "knjižnici", "sustav", "siječanj", "siječnju", "vlak",
        "tisuća", "kemija", "točno", "obitelj", "glazba", "kat",
    ],
    "sr": [
        # ekavian forms: mleko, not mlijeko
        "mleko", "lepo", "lep", "vreme", "pre", "posle", "gde", "nigde",
        "celu", "cela", "ceo", "nedelja", "nedelju", "reči", "reč", "deca",
        "mesto", "mesta",
        # ko / šta (also Bosnian)
        "ko", "neko", "niko", "šta",
        # "da li" questions and "treba da" + verb
        "da li", "treba da",
        # typically Serbian words
        "hleb", "kafa", "pasulj", "mejl", "hiljada", "voz",
    ],
}

# A language must have at least this many MORE marker words than the
# runner-up to win. Higher = more careful, but more "no decision" cases.
MIN_MARKER_LEAD = 1  # LAB 1: change me

# Words that show a post contains English, and words that show it contains BCS.
# If a post has enough of BOTH, we call it "mixed" (code-switching).
# Careful: "i", "a", "to", "on", "me", "no" are words in BOTH languages!
ENGLISH_WORDS = [  # LAB 1: change me
    "the", "is", "was", "are", "and", "of", "for", "in", "my", "so", "it",
    "that", "this", "you", "not", "can't", "i'm", "let's", "how", "what",
    "exam", "passed", "finally", "next", "year", "best", "literally",
    "honestly", "actually", "again", "cold", "food", "done", "thanks", "lot",
    "insane", "swear", "down", "deadline", "website", "free", "asking",
    "friend", "need", "company", "sure", "interesting", "say", "classic",
    "legend", "fire", "assignment", "crashed", "career", "fair",
]
BCS_WORDS = [  # LAB 1: change me
    "je", "sam", "si", "smo", "su", "da", "li", "se", "ne", "nije", "na", "u",
    "za", "od", "iz", "sa", "ko", "šta", "što", "ja", "mi", "ti", "opet", "bio",
    "bila", "bilo", "ima", "nema", "mogu", "danas", "sutra", "kako", "ali",
    "jel", "još", "baš", "ovo", "nešto", "novi",
]

# How many English words AND how many BCS words a post needs to count as "mixed".
MIXED_MIN_WORDS = 2  # LAB 1: change me

# =============================================================================

BCS_LANGUAGES = ["bs", "hr", "sr"]
UNKNOWN = "unknown"

# langdetect is random by default: the same text can get a different answer on
# each run. A fixed seed makes results repeatable - important for a fair comparison.
DetectorFactory.seed = 0

# Building a lingua detector loads language models, which is slow. We build it
# ONCE when this file is imported, and reuse it for every post.
LINGUA_LANGUAGES = {
    Language.BOSNIAN: "bs",
    Language.CROATIAN: "hr",
    Language.SERBIAN: "sr",
    Language.ENGLISH: "en",
}
LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(*LINGUA_LANGUAGES).build()


def detect_naive(text):
    """NLP step: language identification with an off-the-shelf tool (langdetect).

    Why it matters: this is what most people try first. langdetect knows ~55
    languages, including Croatian, but NOT Bosnian or Serbian, so it cannot
    possibly get those right. Expect guesses like Slovenian ("sl") or Slovak ("sk").
    """
    try:
        return detect(text)
    except LangDetectException:
        # Raised when there is nothing to detect, e.g. a post that is only emoji.
        return UNKNOWN


def detect_lingua(text):
    """NLP step: language identification with a restricted candidate list (lingua).

    Why it matters: if we already know which languages to expect, we can tell
    the tool. Choosing from 4 languages is much easier than choosing from 75.
    """
    language = LINGUA_DETECTOR.detect_language_of(text)
    if language is None:
        return UNKNOWN  # lingua was not confident enough to decide
    return LINGUA_LANGUAGES[language]


def tokens_without_diacritics(text):
    """Split a post into words, ignoring diacritics and punctuation."""
    text = strip_diacritics(text.lower())
    words = [word.strip(".,!?:;\"'()…") for word in text.split()]
    return [word for word in words if word]


def count_markers(words, markers):
    """Count how many marker words (or phrases like "da li") occur in the post."""
    # Spaces around the text let us match whole words and phrases with "in":
    # " ko " is found in " ko zna ", but not inside " kolokvij ".
    padded_text = " " + " ".join(words) + " "
    count = 0
    for marker in markers:
        if " " + strip_diacritics(marker) + " " in padded_text:
            count += 1
    return count


def pick_bcs_language(words, fallback):
    """Choose bs, hr or sr by counting marker words for each language."""
    scores = {lang: count_markers(words, MARKER_WORDS[lang]) for lang in BCS_LANGUAGES}
    ranked = sorted(scores, key=scores.get, reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    if scores[winner] - scores[runner_up] >= MIN_MARKER_LEAD:
        return winner
    # Not enough evidence: keep the other detector's guess.
    return fallback


def detect_ours(text):
    """NLP step: language identification with hand-written rules.

    Why it matters: general tools often fail on closely related languages and
    on code-switching. With some linguistic knowledge (ijekavian vs ekavian,
    tko/ko, kruh/hljeb, "je li" vs "da li") we can often do better, and unlike
    a black box, we can see exactly WHY each decision was made.

    Steps:
      1. Enough English AND BCS words -> "mixed" (code-switching).
      2. More English words than BCS words -> "en".
      3. Otherwise ask lingua, and if it says BCS, refine bs/hr/sr with markers.
    """
    words = tokens_without_diacritics(text)
    english_count = count_markers(words, ENGLISH_WORDS)
    bcs_count = count_markers(words, BCS_WORDS)

    if english_count >= MIXED_MIN_WORDS and bcs_count >= MIXED_MIN_WORDS:
        return "mixed"
    if english_count > bcs_count:
        return "en"

    lingua_guess = detect_lingua(text)
    if lingua_guess in BCS_LANGUAGES:
        return pick_bcs_language(words, fallback=lingua_guess)
    return lingua_guess
