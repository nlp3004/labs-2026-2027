"""TEMPLATE for a new dashboard tab. Copy this file to make your own tab.

This file is NOT shown in the dashboard: app.py skips files whose name starts
with "_". To make a tab of your own:

  1. Copy this file and give the copy a name WITHOUT the leading "_",
     e.g. radar/modules/m07_my_tab.py
     (the number decides the order of the tabs: m01, m02, ... m07).
  2. Change NAME to your tab's label.
  3. Write your own code inside render(df).
  4. Restart the dashboard:  uv run streamlit run app.py

You never need to edit app.py. If your tab crashes, only your tab shows the
error message; the other tabs keep working.

-------------------------------------------------------------------------------
What is in df?
-------------------------------------------------------------------------------
df is a pandas DataFrame with ONE ROW PER POST. It is the pipeline's output
(data/processed/radar.csv), so run scripts/run_pipeline.py first.

  Columns every tab can rely on:
    id, source, url, date, title, text   the post itself (title may be empty)
    lang                                 language found by OUR detector

  Extra columns added by the pipeline:
    clean_text, has_diacritics           from radar/clean.py
    lang_naive, lang_lingua, lang_ours   from radar/langid.py (one per detector)
    sent_vader, sent_vader_score         from radar/sentiment.py (VADER: label, -1..+1)
    sent_xlmr, sent_xlmr_score           from radar/sentiment.py (XLM-R: label, confidence 0..1)

  Only after "run_pipeline.py --teaser" (check with: if "cluster" in df.columns):
    cluster                              from radar/embeddings.py (k-means group number)
    map_x, map_y                         from radar/embeddings.py (position on the meaning map)
  The meaning vectors themselves are then in data/processed/embeddings.npy
  (row i = post in row i of df); see radar/modules/m06_meaning.py.

  Gold labels and flags (the human answers, for EVALUATION only):
    lang_gold, sentiment_gold, topic_gold
    ambiguous, sentiment_debatable       True/False flags
    split                                "dev" or "test": report scores on test,
                                         look at mistakes on dev only

Useful Streamlit commands: st.subheader, st.write, st.metric, st.dataframe,
st.plotly_chart, st.selectbox. See https://docs.streamlit.io/develop/api-reference
"""

import plotly.express as px
import streamlit as st

# The label shown on the tab.
NAME = "My new tab"


def render(df):
    """Draw this tab. app.py calls it with the pipeline output (see above)."""
    st.subheader("A number")
    # st.metric shows one big number with a label.
    st.metric("Posts in the corpus", len(df))

    st.subheader("A chart")
    # value_counts() counts how often each value appears. reset_index() turns
    # the result into a table with two columns ("source" and "count"), which
    # is the shape plotly expects.
    per_source = df["source"].value_counts().reset_index()
    st.plotly_chart(px.bar(per_source, x="source", y="count"))

    st.subheader("A table")
    # Pick the columns you want to show; hide_index removes the row numbers.
    st.dataframe(df[["id", "source", "text", "lang"]].head(10), hide_index=True)
