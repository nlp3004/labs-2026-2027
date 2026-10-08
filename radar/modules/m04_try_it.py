"""This tab runs the whole pipeline live on one text you type, and shows the
output of every step. It uses radar/clean.py, radar/langid.py and
radar/sentiment.py - the same functions as scripts/run_pipeline.py."""

import streamlit as st

from radar.clean import clean_text, has_diacritics
from radar.langid import detect_lingua, detect_naive, detect_ours
from radar.sentiment import load_xlmr, sentiment_vader, sentiment_xlmr

NAME = "Try it yourself"


# Loading XLM-R takes a few seconds and ~1 GB of memory. cache_resource loads
# it once and shares it between all re-runs and all browser tabs.
@st.cache_resource
def load_sentiment_model():
    return load_xlmr()


def render(df):
    with st.form("try_it"):
        text = st.text_area("Write a post (any language)", "super, opet nema wifi 🙃")
        submitted = st.form_submit_button("Run the pipeline")
    if not submitted or not text.strip():
        return

    st.subheader("1. Cleaning (radar/clean.py)")
    cleaned = clean_text(text)
    st.code(cleaned, language=None)
    st.write("Contains č ć š ž đ:", has_diacritics(text))

    st.subheader("2. Language identification (radar/langid.py)")
    # Like the pipeline, the detectors read the CLEANED text.
    left, middle, right = st.columns(3)
    left.metric("langdetect", detect_naive(cleaned))
    middle.metric("lingua", detect_lingua(cleaned))
    right.metric("ours", detect_ours(cleaned))

    st.subheader("3. Sentiment (radar/sentiment.py)")
    # Like the pipeline, sentiment reads the ORIGINAL text (capitals, "!!!").
    with st.spinner("Loading XLM-R (only the first time) ..."):
        load_sentiment_model()
    vader_label, vader_score = sentiment_vader(text)
    xlmr_labels, xlmr_confidences = sentiment_xlmr([text])
    left, right = st.columns(2)
    left.metric("VADER", vader_label, f"compound score {vader_score:+.2f}", delta_color="off")
    right.metric("XLM-R", xlmr_labels[0], f"confidence {xlmr_confidences[0]:.0%}",
                 delta_color="off")
