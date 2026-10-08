"""This tab shows how well the three language detectors match the gold labels,
where they get confused, and which posts they get wrong.
It uses radar/langid.py.

Honest evaluation: scores and the confusion table use the TEST half; the list
of mistakes shows DEV posts only, because those are the ones you may tune on."""

import pandas as pd
import plotly.express as px
import streamlit as st

NAME = "Language ID"

# Column in radar.csv -> name shown in the tab
DETECTORS = {
    "lang_naive": "langdetect",
    "lang_lingua": "lingua",
    "lang_ours": "ours",
}

# The confusion table always shows these columns, in this order. Every other
# answer ("sl" = Slovenian, "unknown", ...) is merged into one "other" column.
LANGUAGES = ["bs", "hr", "sr", "en", "mixed"]


def confusion_table(posts, column):
    """Rows: the true language. Columns: what the detector said."""
    predicted = posts[column].where(posts[column].isin(LANGUAGES), "other")
    table = pd.crosstab(posts["lang_gold"], predicted)
    # reindex adds missing rows/columns filled with 0, in our fixed order.
    return table.reindex(index=LANGUAGES, columns=LANGUAGES + ["other"], fill_value=0)


def never_predicted_note(name, table):
    """One sentence about the languages this detector never answers, if any."""
    never = []
    for language in LANGUAGES:
        if table[language].sum() == 0:
            never.append(language)
    if not never:
        return f"{name} predicts every language at least once."
    if len(never) == 1:
        listed = never[0]
    else:
        listed = ", ".join(never[:-1]) + " or " + never[-1]
    return f"{name} never predicts {listed}: every post in those rows must be wrong."


def render(df):
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]
    test_no_mixed = test[test["lang_gold"] != "mixed"]

    st.subheader("Accuracy vs. lang_gold (TEST half)")
    st.caption("'all test posts' is the real score. langdetect and lingua cannot answer "
               "'mixed', so 'without mixed' is a fairer comparison.")
    accuracy = []
    for column, name in DETECTORS.items():
        for label, posts in [("all test posts", test), ("without mixed", test_no_mixed)]:
            correct = posts[column] == posts["lang_gold"]
            accuracy.append({"detector": name, "posts": label, "accuracy": correct.mean()})
    fig = px.bar(pd.DataFrame(accuracy), x="detector", y="accuracy", color="posts",
                 barmode="group", text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig)

    st.subheader("Accuracy per language (TEST half)")
    st.caption("Changed a marker word? Check every column: fixing one language can break another.")
    per_language = {}
    for column, name in DETECTORS.items():
        row = {}
        for language in LANGUAGES:
            posts = test[test["lang_gold"] == language]
            row[language] = (posts[column] == posts["lang_gold"]).mean()
        per_language[name] = row
    table = pd.DataFrame(per_language).T  # .T = transpose: detectors as rows
    st.dataframe(table.style.format("{:.1%}"))

    # index=2 -> "ours" is selected when the tab opens.
    column = st.selectbox(
        "Detector to inspect", list(DETECTORS), index=2, format_func=DETECTORS.get,
        key="language_detector",
    )
    name = DETECTORS[column]

    st.subheader(f"Confusion table: {name} (TEST half)")
    table = confusion_table(test, column)
    st.caption("Rows: the true language. Columns: what the detector said. "
               "'other' = any other language, e.g. Slovenian.")
    st.dataframe(table)
    st.caption(never_predicted_note(name, table))

    st.subheader(f"Posts {name} got wrong (DEV half only)")
    st.caption("Tune your rules on these. Test posts stay hidden, so the test score stays honest.")
    wrong = dev[dev[column] != dev["lang_gold"]]
    languages = st.multiselect(
        "Show only these gold languages", sorted(wrong["lang_gold"].unique()), key="language_filter"
    )
    if languages:
        wrong = wrong[wrong["lang_gold"].isin(languages)]
    search = st.text_input("Search in the text", key="language_search")
    if search:
        wrong = wrong[wrong["text"].str.contains(search, case=False, regex=False)]

    st.write(f"{len(wrong)} posts")
    # The selected detector's answer comes right after the gold label.
    columns = ["id", "text", "lang_gold", column]
    st.dataframe(wrong[columns].rename(columns={column: f"{name} said"}), hide_index=True)
