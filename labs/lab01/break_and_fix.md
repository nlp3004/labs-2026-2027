# Break and fix

**Time:** about 10 minutes. Pick **one** option.

The loop is always the same:

1. Write down the number you want to change (**before**).
2. Change the code.
3. Re-run the pipeline: `uv run python scripts/run_pipeline.py`
4. Refresh the dashboard in your browser (press **R**) and read the number again (**after**).

Use the numbers from the **test** half: the pipeline output and the dashboard both
show them. Look for ideas only in **dev** posts (the *Language ID* tab lists the dev
posts our detector got wrong). Never copy words from test posts, or your test score
stops meaning anything.

---

## Option A: VADER thresholds

In `radar/sentiment.py`, change `POSITIVE_THRESHOLD` and/or `NEGATIVE_THRESHOLD`
(marked `# LAB 1: change me`).

Where to read the result: the pipeline line `VADER (English word list) TEST accuracy`,
or the **VADER test accuracy** number at the top of the dashboard's *Sentiment* tab.

| | POSITIVE_THRESHOLD | NEGATIVE_THRESHOLD | VADER test accuracy |
|---|---|---|---|
| Before | 0.05 | -0.05 | |
| After  |      |       | |

Did it get better? **Why** (or why not)? Hint: look at how many posts have a VADER score
of exactly 0.00, for example in the vote screen or in `data/processed/radar.csv`.

Answer:

---

## Option B: marker words for `detect_ours`

In `radar/langid.py`, add **3 marker words** to `MARKER_WORDS` (to one language or
spread over several). Find them in **dev** posts that `ours` got wrong.

Where to read the result: the table `TEST accuracy per language` in the pipeline output,
or **Accuracy per language (TEST half)** in the dashboard's *Language ID* tab.

Words I added (and to which language):

| | bs | hr | sr | overall test accuracy |
|---|---|---|---|---|
| Before | | | | |
| After  | | | | |

Did fixing one language break another? Why?

Answer:

---

## Option C (harder): a new dashboard tab

Create `radar/modules/m05_stats.py` with `NAME` and `render(df)` that shows **any 3
numbers** about the corpus (e.g. with `st.metric`). Start from `radar/modules/_template.py`.

Check that it passes the module contract:

```bash
uv run pytest
```

Then refresh the dashboard: your tab appears automatically.

My 3 numbers and why I chose them:

Answer:

---

## Commit and push

Commit your change **and** this file with your answers, with a message that says what
you did and what happened, for example:

```bash
git add -A
git commit -m "Lab 1: option B, hr +4.5%"
git push
```
