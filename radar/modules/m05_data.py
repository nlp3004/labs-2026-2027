"""This tab lets you browse the corpus itself: filter the posts by source,
language, missing diacritics or length, and read what the tools had to deal with.
It uses radar/collect.py (the posts) and radar/clean.py (has_diacritics).

Before trusting any accuracy number, LOOK at your data. This tab shows why
this corpus is hard: posts without č ć š ž đ, mixed languages, and posts
so short that almost no tool can decide anything."""

import streamlit as st

NAME = "Data"

BCS_LANGUAGES = ["bs", "hr", "sr"]

# A post with this many words or fewer counts as "very short". Posts like
# "ok", "jok" or "ugh" give a tool almost nothing to work with.
SHORT_POST_WORDS = 3

# Columns shown in the table, in this order.
COLUMNS = ["id", "source", "date", "title", "text", "lang_gold", "sentiment_gold", "topic_gold"]


def word_count(text):
    return len(str(text).split())


def show_difficulty_numbers(df):
    """A few numbers that show why this corpus is hard for NLP tools."""
    bcs = df[df["lang_gold"].isin(BCS_LANGUAGES)]
    no_diacritics = (~bcs["has_diacritics"]).mean()
    mixed = (df["lang_gold"] == "mixed").sum()
    short = (df["text"].apply(word_count) <= SHORT_POST_WORDS).sum()

    first, second, third = st.columns(3)
    first.metric("BCS posts without č ć š ž đ", f"{no_diacritics:.0%}")
    second.metric("Mixed-language posts", mixed)
    third.metric(f"Very short posts (≤ {SHORT_POST_WORDS} words)", short)


def filter_posts(df):
    """Show the filter widgets and return only the posts that match them."""
    # Every widget gets a key starting with "data_". Without it, Streamlit
    # would confuse our "Search in the text" box with the identical one in
    # the Language ID tab, and stop with a "duplicate ID" error.
    left, right = st.columns(2)
    sources = left.multiselect("Source", sorted(df["source"].unique()), key="data_source")
    languages = right.multiselect(
        "True language (lang_gold)", sorted(df["lang_gold"].unique()), key="data_language"
    )
    no_diacritics = st.checkbox(
        "Only BCS posts typed WITHOUT č ć š ž đ", key="data_no_diacritics"
    )
    short_only = st.checkbox(
        f"Only very short posts (≤ {SHORT_POST_WORDS} words)", key="data_short"
    )
    search = st.text_input("Search in the text", key="data_search")

    # Every filter keeps only the matching rows. An empty multiselect
    # means "no filter", not "show nothing".
    if sources:
        df = df[df["source"].isin(sources)]
    if languages:
        df = df[df["lang_gold"].isin(languages)]
    if no_diacritics:
        df = df[df["lang_gold"].isin(BCS_LANGUAGES) & ~df["has_diacritics"]]
    if short_only:
        df = df[df["text"].apply(word_count) <= SHORT_POST_WORDS]
    if search:
        df = df[df["text"].str.contains(search, case=False, regex=False)]
    return df


def render(df):
    st.caption("All posts are SYNTHETIC, about a fictional university. "
               "See data/demo/README.md.")

    st.subheader("Why this data is hard")
    show_difficulty_numbers(df)

    st.subheader("Browse the posts")
    posts = filter_posts(df)
    st.write(f"{len(posts)} of {len(df)} posts")
    st.dataframe(posts[COLUMNS], hide_index=True)
