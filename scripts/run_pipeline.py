"""Run the nlp-radar pipeline (collect -> clean -> language ID -> sentiment).

Run it with:
    uv run python scripts/run_pipeline.py              # the lab: language ID + sentiment
    uv run python scripts/run_pipeline.py --teaser     # + a sneak peek at embeddings

It saves data/processed/radar.csv and prints how well each language detector
and each sentiment model does against the human gold labels (lang_gold,
sentiment_gold). It also saves data/processed/disputed.csv: 15 posts for the
class vote "Who's right, VADER or XLM-R?".

HONEST EVALUATION: every post is in the "dev" half or the "test" half (the
split column). You may look at dev posts and tune your rules on them. The
scores that count are measured on TEST posts, which you never look at while
tuning. That is why the examples printed below come from dev, and the
accuracies from test.

With --teaser, a fifth stage turns every post into a meaning vector
(radar/embeddings.py) and shows what that makes possible: semantic search
across languages, clusters compared to the gold topics, and near-duplicates.

The models must be downloaded first:
    uv run python scripts/download_models.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script import the radar/ package when it is run as a plain file.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from radar.clean import clean_text, has_diacritics  # noqa: E402
from radar.collect import load_corpus  # noqa: E402
from radar.langid import detect_lingua, detect_naive, detect_ours  # noqa: E402
from radar.sentiment import sentiment_vader, sentiment_xlmr  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "radar.csv"
DISPUTED_PATH = REPO_ROOT / "data" / "processed" / "disputed.csv"
# Row i of this array is the meaning vector of row i of radar.csv (--teaser only).
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.npy"

# Column name in the output -> (detector function, short description)
DETECTORS = {
    "lang_naive": (detect_naive, "langdetect (off-the-shelf)"),
    "lang_lingua": (detect_lingua, "lingua (bs/hr/sr/en only)"),
    "lang_ours": (detect_ours, "ours (marker-word rules)"),
}

# Column name in the output -> short description
SENTIMENT_MODELS = {
    "sent_vader": "VADER (English word list)",
    "sent_xlmr": "XLM-R (multilingual transformer)",
}

# The labels we care about. Any other prediction (e.g. "sl" = Slovenian) is "other".
GOLD_LABELS = ["bs", "hr", "sr", "en", "mixed"]
BCS_LABELS = ["bs", "hr", "sr"]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

# The disputed set for the class vote: this many posts of each kind.
DISPUTED_PER_KIND = 5
# Two posts sharing this share of words are variants of the same base post.
NEAR_DUPLICATE_SHARE = 0.6

# Words and emoji that often signal sarcasm: praise that may mean the opposite
# ("Great system." after the portal crashed). A rough guess, used only to
# explain why a post is interesting for the class vote.
SARCASM_CUES = [
    "🙃", "bravo", "divno", "naravno", "lepo", "baš lijepo", "hvala baš",
    "baš vam hvala", "aha", "great", "wonderful", "love that", "thanks a lot",
    "classic", "bold move", "brilliant", "super",
]
EXAMPLES_TO_SHOW = 5

# English queries for the semantic-search teaser. The corpus is mostly BCS, so
# good results here mean the model matched MEANING across languages.
SEARCH_EXAMPLES = [
    "the internet is down",
    "the food in the cafeteria is bad",
    "when are the exam results out?",
]
SEARCH_RESULTS_TO_SHOW = 3
LINE_WIDTH = 72


def print_header(title):
    print()
    print("=" * LINE_WIDTH)
    print(" " + title)
    print("=" * LINE_WIDTH)


# --- Running the stages ------------------------------------------------------

def run_pipeline(stage_count):
    """Collect -> clean -> detect language -> sentiment, with every tool."""
    print(f"[1/{stage_count}] Loading corpus ...", end=" ")
    df = load_corpus()
    print(f"{len(df)} posts")

    print(f"[2/{stage_count}] Cleaning text ...", end=" ")
    df["clean_text"] = df["text"].apply(clean_text)
    df["has_diacritics"] = df["text"].apply(has_diacritics)
    print("done")

    print(f"[3/{stage_count}] Running {len(DETECTORS)} language detectors ...", end=" ", flush=True)
    for column, (detector, _) in DETECTORS.items():
        df[column] = df["clean_text"].apply(detector)
    print("done")

    # The final language used by later stages is OUR detector's answer.
    df["lang"] = df["lang_ours"]

    # Sentiment reads the ORIGINAL text: "GREAT!!!" is stronger than "great",
    # and clean_text would have removed that signal.
    print(f"[4/{stage_count}] Running 2 sentiment models ...", end=" ", flush=True)
    start = time.perf_counter()
    vader_results = df["text"].apply(sentiment_vader)
    df["sent_vader"] = [label for label, _ in vader_results]
    df["sent_vader_score"] = [score for _, score in vader_results]
    vader_seconds = time.perf_counter() - start

    start = time.perf_counter()
    df["sent_xlmr"], df["sent_xlmr_score"] = sentiment_xlmr(df["text"])
    xlmr_seconds = time.perf_counter() - start
    print(f"done (VADER {vader_seconds:.1f} s, XLM-R {xlmr_seconds:.1f} s)")
    return df


def run_embeddings(df):
    """Teaser stage: one meaning vector per post, plus clusters and a 2D map."""
    # Imported here, not at the top: the default run does not need the
    # embedding model at all, not even installed on disk.
    from radar.embeddings import cluster_posts, embed_texts, map_2d

    print("[5/5] Computing sentence embeddings ...", end=" ", flush=True)
    start = time.perf_counter()
    # Embeddings read the ORIGINAL text: the model was trained on normal sentences.
    vectors = embed_texts(df["text"])
    df["cluster"] = cluster_posts(vectors)
    xy = map_2d(vectors)
    df["map_x"] = xy[:, 0]
    df["map_y"] = xy[:, 1]
    print(f"done ({time.perf_counter() - start:.1f} s, {vectors.shape[1]} numbers per post)")
    return vectors


# --- Language ID reports -----------------------------------------------------

def accuracy(df, column):
    """Share of posts where the column equals lang_gold."""
    return (df[column] == df["lang_gold"]).mean()


def confusion_table(gold, predicted):
    """Rows = gold label, columns = predicted label. Unexpected labels -> "other"."""
    predicted = predicted.where(predicted.isin(GOLD_LABELS), "other")
    table = pd.crosstab(gold, predicted, rownames=["gold"], colnames=["predicted"])
    return table.reindex(index=GOLD_LABELS, columns=GOLD_LABELS + ["other"], fill_value=0)


def report_language_summary(dev, test):
    """One table with the numbers that matter: TEST accuracy, with and without "mixed"."""
    test_no_mixed = test[test["lang_gold"] != "mixed"]

    print_header(f"Language ID accuracy - TEST half ({len(test)} posts)")
    print(f"  {'detector':<30}{'test':>8}{'test, no mixed':>16}{'dev':>8}")
    for column, (_, description) in DETECTORS.items():
        print(
            f"  {description:<30}{accuracy(test, column):>8.1%}"
            f"{accuracy(test_no_mixed, column):>16.1%}{accuracy(dev, column):>8.1%}"
        )
    print("  'no mixed': langdetect and lingua cannot answer 'mixed' at all, so")
    print(f"  without those {len(test) - len(test_no_mixed)} posts the comparison is fairer.")
    print("  'dev': for tuning your rules only. The test column is the real score.")

    # Per language: fixing one language can break another, and only a
    # per-language table shows that.
    print("\n  TEST accuracy per language:")
    print(f"  {'detector':<30}", end="")
    for lang in GOLD_LABELS:
        print(f"{lang:>8}", end="")
    print()
    for column, (_, description) in DETECTORS.items():
        print(f"  {description:<30}", end="")
        for lang in GOLD_LABELS:
            print(f"{accuracy(test[test['lang_gold'] == lang], column):>8.1%}", end="")
        print()
    print(f"  {'(posts)':<30}", end="")
    for lang in GOLD_LABELS:
        print(f"{(test['lang_gold'] == lang).sum():>8}", end="")
    print()


def report_detector(test, column, description):
    """Confusion table on the TEST half, and what the "other" answers were."""
    print_header(f"{description} - TEST confusion table")
    print(confusion_table(test["lang_gold"], test[column]).to_string())

    others = test.loc[~test[column].isin(GOLD_LABELS), column].value_counts()
    if len(others) > 0:
        listed = ", ".join(f"{label} x{count}" for label, count in others.items())
        print(f"\n  'other' predictions: {listed}")


def report_ambiguous(test):
    """The posts that are genuinely undecidable between Bosnian and Croatian."""
    ambiguous = test[test["ambiguous"]]
    print_header(f"Ambiguous bs/hr posts in TEST: {len(ambiguous)} (gold 'bs', but 'hr' is defensible)")
    print(f"  {'detector':<30}{'said bs':>9}{'said hr':>9}{'other':>7}")
    for column, (_, description) in DETECTORS.items():
        said_bs = (ambiguous[column] == "bs").sum()
        said_hr = (ambiguous[column] == "hr").sum()
        other = len(ambiguous) - said_bs - said_hr
        print(f"  {description:<30}{said_bs:>9}{said_hr:>9}{other:>7}")


def report_diacritics(test):
    """Do the detectors do worse when BCS posts are typed without diacritics?

    We compare WITHIN each language. One overall "with vs. without" number
    would mix two things: the effect of diacritics, and the fact that the two
    groups contain different shares of bs, hr and sr (which are not equally
    easy to detect). Comparing inside one language keeps the language fixed.
    """
    print_header("BCS posts in TEST: accuracy with vs. without diacritics, per language")
    print(f"  {'':<30}", end="")
    for lang in BCS_LABELS:
        print(f"{lang + ': with':>12}{'without':>9}", end="")
    print()

    for column, (_, description) in DETECTORS.items():
        print(f"  {description:<30}", end="")
        for lang in BCS_LABELS:
            posts = test[test["lang_gold"] == lang]
            with_marks = posts[posts["has_diacritics"]]
            without_marks = posts[~posts["has_diacritics"]]
            print(f"{accuracy(with_marks, column):>12.0%}{accuracy(without_marks, column):>9.0%}", end="")
        print()

    print(f"  {'(posts)':<30}", end="")
    for lang in BCS_LABELS:
        posts = test[test["lang_gold"] == lang]
        with_count = posts["has_diacritics"].sum()
        print(f"{with_count:>12}{len(posts) - with_count:>9}", end="")
    print()
    print("  Compared within one language, because the 'with' and 'without' groups hold")
    print("  different mixes of bs/hr/sr; with 9-31 posts per cell, 1-2 posts = 5-10 points.")


def report_dev_mistakes(dev):
    """Posts OUR detector got wrong - from DEV only, so you can tune on them."""
    wrong = dev[dev["lang_ours"] != dev["lang_gold"]]
    # First one mistake per gold language, then fill up with the next mistakes.
    one_per_language = wrong.groupby("lang_gold", sort=False).head(1)
    examples = pd.concat([one_per_language, wrong.drop(one_per_language.index)])
    examples = examples.head(EXAMPLES_TO_SHOW)

    print_header(f"{len(wrong)} DEV posts our rules got wrong (tune on these), {len(examples)} shown")
    for _, row in examples.iterrows():
        print(f"\n  #{row['id']}: {row['text']}")
        print(
            f"     gold: {row['lang_gold']:<6} ours: {row['lang_ours']:<8}"
            f"lingua: {row['lang_lingua']:<8} langdetect: {row['lang_naive']}"
        )


# --- Sentiment reports -------------------------------------------------------

def report_sentiment_accuracy(test):
    """Accuracy of each sentiment model on TEST: overall and per gold language."""
    print_header(f"Sentiment accuracy - TEST half ({len(test)} posts), by gold language")
    print(f"  {'model':<34}{'overall':>8}", end="")
    for lang in GOLD_LABELS:
        print(f"{lang:>7}", end="")
    print()

    for column, description in SENTIMENT_MODELS.items():
        correct = test[column] == test["sentiment_gold"]
        print(f"  {description:<34}{correct.mean():>8.1%}", end="")
        for lang in GOLD_LABELS:
            in_lang = test["lang_gold"] == lang
            print(f"{correct[in_lang].mean():>7.0%}", end="")
        print()

    counts = test["lang_gold"].value_counts()
    listed = ", ".join(f"{lang} {counts[lang]}" for lang in GOLD_LABELS)
    print(f"  (posts: {listed})")


def precision_recall_f1(gold, predicted, label):
    """Precision, recall and F1 for ONE label, e.g. "negative".

    precision: of the posts the model CALLED negative, how many really are?
    recall:    of the posts that really ARE negative, how many did it find?
    F1:        one number that is high only if BOTH are high.

    Accuracy alone can hide a bad model: always saying "negative" gets a
    decent accuracy here, but its recall for "positive" and "neutral" is 0.
    """
    called_label = predicted == label
    really_label = gold == label
    # Posts that are this label AND were called this label.
    found = (called_label & really_label).sum()

    # Guard against dividing by zero: a model that never says "neutral"
    # has no neutral predictions to be precise about.
    precision = 0.0
    if called_label.sum() > 0:
        precision = found / called_label.sum()

    recall = 0.0
    if really_label.sum() > 0:
        recall = found / really_label.sum()

    f1 = 0.0
    if precision + recall > 0:
        # The harmonic mean punishes imbalance: precision 1.0 with recall 0.1
        # gives F1 0.18, not the 0.55 a normal average would give.
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def macro_f1(gold, predicted):
    """The average F1 over all sentiment labels, each label counting equally.

    "Macro" means a small class matters as much as a big one. A model must
    do well on ALL labels to score high.
    """
    total = 0.0
    for label in SENTIMENT_LABELS:
        _, _, f1 = precision_recall_f1(gold, predicted, label)
        total += f1
    return total / len(SENTIMENT_LABELS)


def report_sentiment_metrics(test):
    """Confusion table and per-label precision / recall / F1 for each model, on TEST."""
    gold = test["sentiment_gold"]

    for column, description in SENTIMENT_MODELS.items():
        predicted = test[column]
        model_accuracy = (predicted == gold).mean()
        print_header(
            f"{description:<34} TEST accuracy: {model_accuracy:5.1%}   "
            f"macro-F1: {macro_f1(gold, predicted):.2f}"
        )

        table = pd.crosstab(gold, predicted, rownames=["gold"], colnames=["predicted"])
        table = table.reindex(index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS, fill_value=0)
        print(table.to_string())

        print(f"\n  {'label':<10}{'precision':>11}{'recall':>9}{'F1':>7}{'posts':>8}")
        for label in SENTIMENT_LABELS:
            precision, recall, f1 = precision_recall_f1(gold, predicted, label)
            posts = (gold == label).sum()
            print(f"  {label:<10}{precision:>11.2f}{recall:>9.2f}{f1:>7.2f}{posts:>8}")

    # Always answering the most common label is the baseline to beat. Its
    # accuracy can look respectable; macro-F1 shows it is useless for 2 of 3 labels.
    most_common = gold.value_counts().index[0]
    always_most_common = pd.Series(most_common, index=gold.index)
    baseline_accuracy = (always_most_common == gold).mean()
    baseline_f1 = macro_f1(gold, always_most_common)
    print(
        f"\n  Baseline - always say '{most_common}': "
        f"accuracy {baseline_accuracy:.1%}, macro-F1 {baseline_f1:.2f}"
    )


def report_disagreement(test):
    """How often do VADER and XLM-R give different labels (TEST half)?"""
    disagree = test["sent_vader"] != test["sent_xlmr"]
    print_header(
        f"VADER and XLM-R disagree on {disagree.sum()}/{len(test)} TEST posts ({disagree.mean():.1%})"
    )
    table = pd.crosstab(test["sent_vader"], test["sent_xlmr"], rownames=["VADER"], colnames=["XLM-R"])
    print(table.to_string())

    # When they disagree, who is right?
    disputed = test[disagree]
    vader_right = (disputed["sent_vader"] == disputed["sentiment_gold"]).sum()
    xlmr_right = (disputed["sent_xlmr"] == disputed["sentiment_gold"]).sum()
    neither = len(disputed) - vader_right - xlmr_right
    print(f"\n  On those posts: VADER right {vader_right}, XLM-R right {xlmr_right}, neither {neither}")


# --- The disputed set for the class vote -------------------------------------

def has_emoji(text):
    """Rough check: emoji live above U+2600 in Unicode; letters like č ć š ž đ do not."""
    return any(ord(char) >= 0x2600 for char in text)


def padded_words(text):
    """Lowercase words without punctuation, with a space on each side.

    The spaces let us match whole words with "in": " aha " is found in
    " aha super ", but not inside " haha ".
    """
    words = [word.strip(".,!?:;\"'()…") for word in text.lower().split()]
    return " " + " ".join(words) + " "


def has_sarcasm_cue(text):
    text = padded_words(text)
    for cue in SARCASM_CUES:
        if " " + cue + " " in text:
            return True
    return False


def is_near_duplicate(text, chosen_texts):
    """Is this post a variant (typo, no diacritics) of one we already chose?"""
    words = set(padded_words(text).split())
    for other in chosen_texts:
        other_words = set(padded_words(other).split())
        shared = len(words & other_words) / min(len(words), len(other_words))
        if shared >= NEAR_DUPLICATE_SHARE:
            return True
    return False


def interesting_reasons(row):
    """List what makes a post worth discussing in class."""
    reasons = []
    if has_sarcasm_cue(row["text"]):
        reasons.append("sarcasm?")
    if has_emoji(row["text"]):
        reasons.append("emoji")
    if "..." in row["text"] or "…" in row["text"]:
        reasons.append("trailing off...")
    if row["lang_gold"] in BCS_LABELS and not row["has_diacritics"]:
        reasons.append("no diacritics")
    if row["lang_gold"] == "mixed":
        reasons.append("mixed language")
    if row["sent_vader_score"] == 0:
        # A score of exactly 0 usually means "VADER knew none of these words".
        reasons.append("VADER knew no words")
    else:
        reasons.append("VADER found opinion words")
    return reasons


def pick_kind(candidates, kind, chosen_texts):
    """The DISPUTED_PER_KIND most interesting candidates, skipping near-duplicates.

    DEV posts come first. Only if dev has too few of this kind do we take
    test posts, and we say so: showing their gold label in class means
    they are no longer unseen.
    """
    candidates = candidates.copy()
    candidates["from_dev"] = candidates["split"] == "dev"
    candidates["reasons"] = candidates.apply(interesting_reasons, axis=1)
    candidates["interest"] = candidates["reasons"].apply(len)
    # Tie-breaker: both models sure of themselves.
    candidates["confidence"] = candidates["sent_vader_score"].abs() + candidates["sent_xlmr_score"]
    candidates = candidates.sort_values(["from_dev", "interest", "confidence"], ascending=False)

    picked = []
    for _, row in candidates.iterrows():
        if len(picked) == DISPUTED_PER_KIND:
            break
        if is_near_duplicate(row["text"], chosen_texts):
            continue
        chosen_texts.append(row["text"])
        reasons = row["reasons"]
        if not row["from_dev"]:
            reasons = reasons + ["(test post)"]
        row["why_interesting"] = kind + ": " + ", ".join(reasons)
        picked.append(row)
    return picked


def pick_disputed(df):
    """15 posts for the class vote, from DEV wherever possible:

      - 5 where VADER is right and XLM-R is wrong,
      - 5 where XLM-R is right and VADER is wrong,
      - 5 where the gold label itself is debatable (sarcasm, understatement).
    """
    disagree = df["sent_vader"] != df["sent_xlmr"]
    clear = ~df["sentiment_debatable"]
    vader_wins = df[disagree & clear & (df["sent_vader"] == df["sentiment_gold"])]
    xlmr_wins = df[disagree & clear & (df["sent_xlmr"] == df["sentiment_gold"])]
    # Debatable posts: those the models disagree on are the most fun to vote on.
    debatable = df[df["sentiment_debatable"] & disagree]

    chosen_texts = []
    rows = []
    rows += pick_kind(vader_wins, "VADER wins", chosen_texts)
    rows += pick_kind(xlmr_wins, "XLM-R wins", chosen_texts)
    rows += pick_kind(debatable, "gold debatable", chosen_texts)

    # sentiment_gold comes last: hide that column until after the vote!
    columns = [
        "id", "lang_gold", "text", "sent_vader", "sent_vader_score",
        "sent_xlmr", "sent_xlmr_score", "why_interesting", "sentiment_gold",
    ]
    return pd.DataFrame(rows)[columns]


def report_disputed(disputed):
    print_header(f"{len(disputed)} disputed posts for the vote: who's right, VADER or XLM-R?")
    for _, row in disputed.iterrows():
        print(f"\n  #{row['id']} [{row['lang_gold']}] {row['text']}")
        print(
            f"     VADER: {row['sent_vader']:<8} ({row['sent_vader_score']:+.2f})   "
            f"XLM-R: {row['sent_xlmr']:<8} ({row['sent_xlmr_score']:.0%})   "
            f"gold: {row['sentiment_gold']}"
        )
        print(f"     [{row['why_interesting']}]")


# --- Teaser reports (--teaser only) ------------------------------------------

def report_semantic_search(df, vectors):
    """Search the corpus with English questions: do we find BCS posts?"""
    from radar.embeddings import semantic_search

    print_header("TEASER - Semantic search: English queries, posts in any language")
    for query in SEARCH_EXAMPLES:
        rows, scores = semantic_search(query, vectors, SEARCH_RESULTS_TO_SHOW)
        print(f"\n  Query: \"{query}\"")
        for row, score in zip(rows, scores):
            post = df.iloc[row]
            print(f"    {score:.2f}  [{post['lang_gold']}] {post['text']}")


def purity(clusters, gold):
    """Share of posts whose cluster's most common gold label is also theirs.

    1.0 = every cluster contains only one gold label. It is a simple way to
    check clusters against labels the clustering never saw.
    """
    table = pd.crosstab(clusters, gold)
    return table.max(axis=1).sum() / len(gold)


def report_clusters(df):
    """Do the clusters (found WITHOUT labels) match the gold topics?"""
    print_header("TEASER - Clusters (k-means on embeddings) vs. topic_gold")
    print("  Rows: cluster number (made up by k-means). Columns: the gold topic.")
    print(pd.crosstab(df["cluster"], df["topic_gold"]).to_string())

    # Compare with language: do clusters follow WHAT a post says, or the
    # language it is written in?
    print(f"\n  Topic purity:    {purity(df['cluster'], df['topic_gold']):.0%}"
          "   (do clusters match the gold topics?)")
    print(f"  Language purity: {purity(df['cluster'], df['lang_gold']):.0%}"
          "   (do clusters match the languages?)")


def report_near_duplicates(df, vectors):
    """How many posts are (almost) copies of each other?"""
    from radar.embeddings import DUPLICATE_THRESHOLD, near_duplicate_pairs

    pairs = near_duplicate_pairs(vectors)
    in_a_pair = set()
    for a, b, _ in pairs:
        in_a_pair.add(a)
        in_a_pair.add(b)
    print_header(
        f"TEASER - Near-duplicates (similarity >= {DUPLICATE_THRESHOLD:.2f}): "
        f"{len(pairs)} pairs, {len(in_a_pair)} of {len(df)} posts"
    )
    # The most similar pairs are often boring (one changed letter), so we also
    # show pairs from just above the threshold, where it gets interesting.
    examples = pairs[:2] + pairs[-2:]
    for a, b, score in examples:
        print(f"\n  {score:.2f}  #{df['id'][a]}: {df['text'][a]}")
        print(f"        #{df['id'][b]}: {df['text'][b]}")


# --- Main --------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the nlp-radar pipeline.")
    parser.add_argument(
        "--teaser", action="store_true",
        help="also run the sentence-embedding stage (semantic search, clusters, near-duplicates)",
    )
    return parser.parse_args()


def main():
    # Windows terminals may not print emoji or č ć š ž đ without this.
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    stage_count = 5 if arguments.teaser else 4

    title = "language ID + sentiment"
    if arguments.teaser:
        title += " + embeddings teaser"
    print_header(f"nlp-radar: {title} on SYNTHETIC demo posts")

    pipeline_start = time.perf_counter()
    df = run_pipeline(stage_count)
    vectors = None
    if arguments.teaser:
        vectors = run_embeddings(df)
    print(f"Pipeline took {time.perf_counter() - pipeline_start:.1f} s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    if arguments.teaser:
        np.save(EMBEDDINGS_PATH, vectors)
        print(f"Saved {EMBEDDINGS_PATH.relative_to(REPO_ROOT)}")
    elif EMBEDDINGS_PATH.exists():
        # Left over from an earlier --teaser run; it would no longer match radar.csv.
        EMBEDDINGS_PATH.unlink()

    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]

    report_language_summary(dev, test)
    for column, (_, description) in DETECTORS.items():
        report_detector(test, column, description)
    report_ambiguous(test)
    report_diacritics(test)
    report_dev_mistakes(dev)

    report_sentiment_accuracy(test)
    report_sentiment_metrics(test)
    report_disagreement(test)

    disputed = pick_disputed(df)
    disputed.to_csv(DISPUTED_PATH, index=False)
    report_disputed(disputed)
    print(f"\nSaved {DISPUTED_PATH.relative_to(REPO_ROOT)}")

    if arguments.teaser:
        report_semantic_search(df, vectors)
        report_clusters(df)
        report_near_duplicates(df, vectors)
    else:
        print("\nWant a sneak peek at sentence embeddings? Run it again with --teaser.")
    print()


if __name__ == "__main__":
    main()
