"""This tab compares the two sentiment models (VADER and XLM-R) per language,
and shows the posts they disagree on, one at a time, for the class vote.
It uses radar/sentiment.py."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

NAME = "Sentiment"

DISPUTED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "disputed.csv"
MODELS = {"sent_vader": "VADER", "sent_xlmr": "XLM-R"}
VOTE_CHOICES = ["VADER", "XLM-R", "neither"]


def votes_for(post_id):
    """The vote counts for one post, kept in st.session_state.

    st.session_state survives Streamlit's re-runs (every click is a re-run),
    so the counts are remembered while the dashboard stays open. They reset
    when the page is reloaded.
    """
    if "votes" not in st.session_state:
        st.session_state.votes = {}
    if post_id not in st.session_state.votes:
        st.session_state.votes[post_id] = {"VADER": 0, "XLM-R": 0, "neither": 0}
    return st.session_state.votes[post_id]


def render(df):
    test = df[df["split"] == "test"]

    st.subheader("Accuracy vs. sentiment_gold (TEST half)")
    metric_columns = st.columns(len(MODELS))
    for metric_column, (column, model) in zip(metric_columns, MODELS.items()):
        model_accuracy = (test[column] == test["sentiment_gold"]).mean()
        metric_column.metric(f"{model} test accuracy", f"{model_accuracy:.1%}")

    st.subheader("Accuracy by language (TEST half)")
    rows = []
    for column, model in MODELS.items():
        correct = test[column] == test["sentiment_gold"]
        for lang, lang_correct in correct.groupby(test["lang_gold"]):
            rows.append({"model": model, "language": lang, "accuracy": lang_correct.mean()})
    fig = px.bar(pd.DataFrame(rows), x="language", y="accuracy", color="model",
                 barmode="group", text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig)

    agree = (test["sent_vader"] == test["sent_xlmr"]).mean()
    st.metric("VADER and XLM-R agree on", f"{agree:.0%} of test posts")

    st.subheader("Disputed posts: who's right, VADER or XLM-R?")
    if not DISPUTED_PATH.exists():
        st.info("Run: uv run python scripts/run_pipeline.py")
        return
    disputed = pd.read_csv(DISPUTED_PATH)

    # st.session_state survives Streamlit's re-runs, so it remembers which
    # post we are on. A normal variable would reset on every click.
    if "disputed_index" not in st.session_state:
        st.session_state.disputed_index = 0
    post = disputed.iloc[st.session_state.disputed_index % len(disputed)]

    st.caption(f"Post {st.session_state.disputed_index % len(disputed) + 1} of {len(disputed)}"
               f"  ·  language: {post['lang_gold']}")
    st.markdown(f"### {post['text']}")
    left, right = st.columns(2)
    left.metric("VADER says", post["sent_vader"], f"score {post['sent_vader_score']:+.2f}",
                delta_color="off")
    right.metric("XLM-R says", post["sent_xlmr"], f"confidence {post['sent_xlmr_score']:.0%}",
                 delta_color="off")

    # The class vote: one click per raised hand. The buttons come BEFORE the
    # expander, so a click is already counted when the counts are shown below.
    post_id = int(post["id"])
    votes = votes_for(post_id)
    st.write("**Who's right?** Click once per vote.")
    vote_columns = st.columns(len(VOTE_CHOICES))
    for vote_column, choice in zip(vote_columns, VOTE_CHOICES):
        if vote_column.button(choice, key=f"vote_{choice}_{post_id}", use_container_width=True):
            votes[choice] += 1
    total = sum(votes.values())
    st.caption(f"{total} votes so far (the counts are shown after the reveal)")

    # Hidden until after the vote. The expander closes again on the next post.
    with st.expander("Reveal the gold label", key=f"reveal_{post_id}"):
        st.write(f"**{post['sentiment_gold']}**  ·  why this post: {post['why_interesting']}")
        st.write(
            f"Class votes: VADER **{votes['VADER']}** · XLM-R **{votes['XLM-R']}** · "
            f"neither **{votes['neither']}**"
        )

    if st.button("Next ▶"):
        st.session_state.disputed_index += 1
        st.rerun()
