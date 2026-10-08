"""This tab shows the corpus at a glance: how many posts, where they come from,
which languages they are in, and how sentiment changes month by month.
It uses radar/collect.py (the posts), radar/langid.py (detected language)
and radar/sentiment.py (XLM-R sentiment)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from radar.modules._colors import SENTIMENT_COLORS, SENTIMENT_ORDER

NAME = "Overview"


def render(df):
    left, right = st.columns(2)
    left.metric("Posts", len(df))
    right.metric("Sources", df["source"].nunique())

    st.subheader("Posts per source")
    per_source = df["source"].value_counts().reset_index()
    st.plotly_chart(px.bar(per_source, x="source", y="count"))

    st.subheader("Languages: gold label vs. our detector")
    # Two columns side by side -> one long table with a "label type" column,
    # which is the shape plotly needs for grouped bars.
    gold = df["lang_gold"].value_counts().rename("gold (lang_gold)")
    detected = df["lang"].value_counts().rename("detected (lang)")
    languages = pd.concat([gold, detected], axis=1).fillna(0).reset_index()
    languages = languages.melt(id_vars="index", var_name="label type", value_name="posts")
    fig = px.bar(languages, x="index", y="posts", color="label type", barmode="group")
    fig.update_layout(xaxis_title="language")
    st.plotly_chart(fig)

    st.subheader("Sentiment over time (XLM-R), by month")
    st.caption(
        "Three events are hidden in this chart: the January exam period, a wifi outage "
        "in March, and Lipa Fest in May. Can you find them? Which sentiment gives each "
        "one away?"
    )
    # "2025-09-17" -> "2025-09", so we can count posts per month.
    month = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    per_month = df.groupby([month, "sent_xlmr"]).size().reset_index(name="posts")
    fig = px.bar(
        per_month, x="date", y="posts", color="sent_xlmr",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sent_xlmr": SENTIMENT_ORDER},
        labels={"date": "month", "sent_xlmr": "sentiment"},
    )
    st.plotly_chart(fig)

    # The same numbers as a table: readable without telling colours apart.
    with st.expander("Show as a table"):
        table = per_month.pivot(index="date", columns="sent_xlmr", values="posts")
        table = table.reindex(columns=SENTIMENT_ORDER).fillna(0).astype(int)
        st.dataframe(table)
