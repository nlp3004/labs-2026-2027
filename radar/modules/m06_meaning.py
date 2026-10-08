"""This tab shows what sentence embeddings can do: search posts by MEANING in
any language, see every post on a map of meanings, and find near-duplicates.
It uses radar/embeddings.py and data/processed/embeddings.npy (one meaning
vector per post, saved by scripts/run_pipeline.py --teaser)."""

from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

from radar.embeddings import DUPLICATE_THRESHOLD, load_embedder, near_duplicate_pairs, semantic_search

NAME = "Search by meaning"

EMBEDDINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings.npy"
SEARCH_RESULTS = 10
DUPLICATE_PAIRS_TO_SHOW = 30

# What to colour the map by: column -> label in the dropdown.
MAP_COLOURS = {
    "topic_gold": "true topic (topic_gold)",
    "lang_gold": "true language (lang_gold)",
    "cluster": "k-means cluster",
}


# Like the CSV in app.py: read the file once, and again only when it changes.
@st.cache_data
def load_vectors(last_modified):
    return np.load(EMBEDDINGS_PATH)


# Loading the model takes a few seconds. cache_resource loads it once and
# shares it between all re-runs and all browser tabs.
@st.cache_resource
def load_embedding_model():
    return load_embedder()


def show_search(df, vectors):
    st.subheader("Search by meaning")
    st.caption("Try an English query: the best matches are often in Bosnian, Croatian or Serbian.")
    query = st.text_input("Query", "the internet is down", key="meaning_query")
    if not query.strip():
        return

    with st.spinner("Loading the embedding model (only the first time) ..."):
        load_embedding_model()
    rows, scores = semantic_search(query, vectors, SEARCH_RESULTS)

    results = df.iloc[rows][["id", "text", "lang_gold", "topic_gold"]].copy()
    results.insert(0, "similarity", scores)
    st.dataframe(results, hide_index=True, column_config={
        "similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
    })


def show_map(df):
    st.subheader("A map of meanings")
    st.caption(
        "Every dot is a post. Each post's 384 numbers are squeezed down to 2, so "
        "posts close together usually mean similar things. Hover a dot to read it."
    )
    colour = st.selectbox(
        "Colour the dots by", list(MAP_COLOURS), format_func=MAP_COLOURS.get, key="meaning_colour"
    )
    # Cluster numbers are names, not amounts: as text, plotly gives each its
    # own colour instead of a scale from light to dark.
    colours = df[colour].astype(str)
    fig = px.scatter(
        df, x="map_x", y="map_y", color=colours, hover_data=["id", "text", "lang_gold"],
        labels={"color": MAP_COLOURS[colour]},
    )
    # The axes have no meaning of their own, only distances do.
    fig.update_xaxes(showticklabels=False, title=None)
    fig.update_yaxes(showticklabels=False, title=None)
    st.plotly_chart(fig)


def show_near_duplicates(df, vectors):
    st.subheader("Near-duplicates")
    threshold = st.slider(
        "Similarity threshold", 0.70, 1.00, DUPLICATE_THRESHOLD, 0.01, key="meaning_threshold"
    )
    pairs = near_duplicate_pairs(vectors, threshold)
    st.write(f"{len(pairs)} pairs of posts with similarity ≥ {threshold:.2f}. "
             "Lower the threshold: at what point do the pairs stop being copies?")

    # Show the pairs closest to the threshold: that is where it gets interesting.
    rows = []
    for a, b, score in reversed(pairs[-DUPLICATE_PAIRS_TO_SHOW:]):
        rows.append({"similarity": score, "post A": df["text"][a], "post B": df["text"][b]})
    st.dataframe(rows, hide_index=True)


def render(df):
    if not EMBEDDINGS_PATH.exists() or "cluster" not in df.columns:
        st.info(
            "This tab is a teaser for later labs. To fill it, run:  "
            "uv run python scripts/run_pipeline.py --teaser"
        )
        return
    vectors = load_vectors(EMBEDDINGS_PATH.stat().st_mtime)
    if len(vectors) != len(df):
        st.warning("embeddings.npy and radar.csv do not match. Re-run the pipeline.")
        return

    show_search(df, vectors)
    show_map(df)
    show_near_duplicates(df, vectors)
