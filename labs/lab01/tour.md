# Code tour: find your way around NLP Radar

**Time:** about 20 minutes. Work in pairs.

**Before you start**, run the pipeline once and open the dashboard:

```bash
uv run python scripts/run_pipeline.py
uv run streamlit run app.py
```

Every answer is in the **code** or the **dashboard**: you cannot answer these questions
from general NLP knowledge alone. Write file names and line numbers like this:
`radar/clean.py:42`. Tip: in VS Code, `Ctrl+P` opens a file by name and `Ctrl+G`
jumps to a line.

---

## FIND: where does it happen?

**1.** The pipeline gives every post a final language label in the `lang` column.
Which **file and function** decides that label, and on which **line** is the `lang`
column filled in? (Hint: there are three detectors, but only one of them "wins".)

Answer:

**2.** Where does a new dashboard tab come from? Find the line in `app.py` that
discovers the tab files. Which **two things** must every tab file define, and how does
`app.py` use each of them?

Answer:

**3.** VADER turns a score between -1 and +1 into `positive`, `neutral` or `negative`.
Where are the two **thresholds** set (file + lines), and in which function are they used?

Answer:

---

## TRACE: follow one post through the pipeline

Find each post in the dashboard: the **Language ID** tab lists the dev posts our
detector got wrong, and the **Try it yourself** tab runs every stage on any text you
paste in. You can also open `data/processed/radar.csv`.

**4.** Take post **#13**: `super, opet nema wifi 🙃` (gold label: Bosnian, negative).
Write what each stage outputs for it:

| Stage                            | Output |
|----------------------------------|--------|
| cleaned text (`clean.py`)        |        |
| langdetect                       |        |
| lingua                           |        |
| ours (`detect_ours`)             |        |
| VADER label + score              |        |
| XLM-R label + confidence         |        |

Which stages got it wrong?

Answer:

**5.** Post **#47**, `studentska služba lost my papers, opet moram sve ispočetka`, is a
**mixed** post, but `detect_ours` says `hr`. Read `detect_ours` in `radar/langid.py`
step by step: how many words does it find in `ENGLISH_WORDS`, and how many in `BCS_WORDS`?
Which step of the function makes the final decision, and why is it not "mixed"?

Answer:

**6.** Post **#50**, `passed sam!!! finally mogu spavat 😭`, is **positive** (the student
passed and can finally sleep). VADER says **negative**. Use the *Try it yourself* tab:
paste the post, then paste it again **without** the emoji. What is VADER's score each
time? Which single part of the post makes VADER negative, and what does XLM-R say?

Answer:

---

## PREDICT: guess first, then check

Write your guess **before** you change anything. Then make the change, re-run the
pipeline, refresh the dashboard, and write down what really happened.
**Undo your change afterwards** (`git checkout radar/langid.py`).

**7.** In `radar/langid.py`, delete the word `"tjedan"` from the `"hr"` list in
`MARKER_WORDS`. Will `ours` get **better, worse or the same** on Croatian (`hr`) test posts?

My guess (before):

What really happened (after), and why:

**8.** In `radar/langid.py`, change `MIXED_MIN_WORDS` from `2` to `1`. Which languages'
test accuracy will change, and in which direction? Will any language **lose** posts to
"mixed"?

My guess (before):

What really happened (after), and why:
