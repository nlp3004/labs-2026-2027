# nlp-radar

Starter repo for **Lab 1 of IT 3004 Introduction to NLP**.

nlp-radar is a small NLP pipeline that works like a "radar" for student opinion.
It reads social-media posts about a university and answers two questions:

1. **Which language is each post written in?** Bosnian, Croatian, Serbian, English, or a mix.
2. **What is its sentiment?** Positive, negative, or neutral.

It then compares several tools against human labels, **honestly** (on posts the tools
were never tuned on), and shows the results in an interactive dashboard.

An optional **teaser** for later labs (`--teaser`) adds a third question: **what does
each post MEAN?** Every post becomes a vector of numbers (an *embedding*), so we can
search posts by meaning across languages, group them into topics without any labels,
and find near-duplicate posts.

> ⚠️ **All data in this repo is SYNTHETIC.** The posts are about *Univerzitet Lipa*,
> a **fictional** university in the **fictional** city of Brezovac. No real people,
> brands, or institutions appear in it. See [`data/demo/README.md`](data/demo/README.md).

---

## What you will learn

- What a typical **NLP pipeline** looks like: collect → clean → identify language → analyse
  sentiment.
- Why **language identification** is hard for closely related languages (Bosnian, Croatian
  and Serbian share most of their words) and for posts that mix two languages.
- The difference between three kinds of NLP tools:
  - **hand-written rules** (our own marker-word language detector),
  - **off-the-shelf libraries** (`langdetect`, `lingua`, VADER),
  - **machine-learning models** (XLM-R, a multilingual transformer).
- How to **evaluate** tools against human "gold" labels **honestly**: tune on a *dev*
  half, report on a *test* half you never looked at, and why a single accuracy number
  never tells the whole story.
- *Teaser:* how **sentence embeddings** represent meaning as numbers, and what that makes
  possible: cross-lingual semantic search, **unsupervised** clustering (k-means), and
  near-duplicate detection.

---

## Setup (do this once, with internet)

### 1. Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and all packages.
You do **not** need to install Python yourself: uv downloads the right version (3.12).

- **Windows** (PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Close and reopen your terminal, then check that it works: `uv --version`

### 2. Clone the repo and install the packages

```bash
git clone <repo-url>
cd nlp-radar
uv sync
```

`uv sync` creates a `.venv/` folder with Python 3.12 and the exact package versions
listed in `uv.lock`. It installs the **CPU-only** version of PyTorch, because this lab
does not need a GPU.

### 3. Download the model (~1 GB)

```bash
uv run python scripts/download_models.py
```

This saves the sentiment model to `models/`. For the optional embedding teaser, add
`--teaser` to also download a second model:

| Folder                        | Model                                    | Size    | Used by               | Download with |
|-------------------------------|------------------------------------------|---------|-----------------------|---------------|
| `models/xlmr-sentiment/`      | XLM-R, fine-tuned for sentiment          | ~1 GB   | `radar/sentiment.py`  | always        |
| `models/multilingual-minilm/` | Multilingual MiniLM, sentence embeddings | ~470 MB | `radar/embeddings.py` | `--teaser`    |

It is the **only** step that needs the internet. Use a good connection, since it can take a
few minutes. A model that is already downloaded is skipped, so it is safe to run again.

### 4. Check your setup

```bash
uv run python check_setup.py
```

This checks the Python version, the uv environment, every package, the XLM-R model (and
that it can really classify a sentence), the demo corpus, the pipeline output and your git
settings. Run it **while you still have internet**, so you can fix anything that is missing.
Every line gets a ✅ or a ❌ (`[OK]` / `[X]` on terminals without emoji), and every ❌
comes with the exact command that fixes it. Fix problems **top to bottom**: a later
check often fails only because an earlier one did. On a fresh clone, "Pipeline output"
fails until you run the pipeline once (Step 1 below).

✅ **After these steps, everything runs offline.**

---

## Running the project

Always run commands through `uv run`, so they use the project's Python and packages.

### Step 1: Run the pipeline

```bash
uv run python scripts/run_pipeline.py
```

This runs the four lab stages on the 300 demo posts and prints a report. All scores
are measured on the **test half** (see [Honest evaluation](#honest-evaluation-dev-and-test) below):

- a summary table: accuracy of each language detector on test, on test **without the
  "mixed" posts** (which langdetect and lingua cannot answer at all), and on dev,
- a confusion table for each language detector,
- how the detectors handle the 10 posts that are **ambiguous** between Bosnian and Croatian,
- how much worse the detectors do on posts typed **without diacritics** (č ć š ž đ),
- **dev** posts our own detector got wrong, so you can improve its rules,
- sentiment accuracy per language for VADER and XLM-R,
- for each sentiment model: a confusion table, precision / recall / F1 per label, and
  macro-F1, compared to a simple baseline,
- how often VADER and XLM-R **disagree**,
- 15 posts for a class vote: 5 where VADER is right, 5 where XLM-R is right, and 5
  where the gold label itself is debatable.

It saves two files:

| File                           | Contents                                                     |
|--------------------------------|--------------------------------------------------------------|
| `data/processed/radar.csv`     | Every post with all predictions added                        |
| `data/processed/disputed.csv`  | The 15 posts for the class vote, with a `why_interesting` column |

#### Optional: the embeddings teaser

```bash
uv run python scripts/download_models.py --teaser    # once, needs internet
uv run python scripts/run_pipeline.py --teaser
```

This adds a fifth stage (`radar/embeddings.py`) and three more report sections:
**semantic search** with English queries, the **clusters** found by k-means compared to the
gold topics and to the languages, and **near-duplicate** posts. It also saves
`data/processed/embeddings.npy` (one meaning vector of 384 numbers per post) and fills the
dashboard's *Search by meaning* tab.

### Honest evaluation: dev and test

Every post has a `split` column: half are **dev**, half are **test** (50/50 for every language).

- **dev**: look at these posts, and tune your rules on them (e.g. add marker words to
  `detect_ours` because of a dev post it got wrong).
- **test**: only for the final score. Do **not** read test posts while tuning.

Why? If you add a marker word because you saw it in a post, your detector will get
*that* post right, and a score measured on the same posts would look better than the
detector really is. A score on posts it has never seen tells you how it will do on new
posts. That is why the report and the dashboard show **mistakes from dev** and **scores from test**.

### Step 2: Open the dashboard

```bash
uv run streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Stop the dashboard with `Ctrl+C` in the terminal.

| Tab                 | What it shows                                                                  |
|---------------------|--------------------------------------------------------------------------------|
| **Overview**        | Posts per source, detected vs. true languages, sentiment month by month        |
| **Language ID**     | Test accuracy of the 3 detectors (with and without "mixed"), a test confusion table, and a searchable list of **dev** mistakes |
| **Sentiment**       | VADER vs. XLM-R test accuracy per language, and the class vote: *"Who's right?"* |
| **Try it yourself** | Type any text and watch every pipeline step process it live                     |
| **Data**            | Browse the posts, filtered by source, language, missing diacritics or length |
| **Search by meaning** | *Teaser* (after a `--teaser` run): search posts by meaning in any language, a map of all posts, and near-duplicates |

> The dashboard reads `data/processed/radar.csv` and reloads it whenever the file changes.
> If you change the code, **re-run the pipeline** (Step 1), then **refresh the browser**
> (press **R**) to see the new results. No need to restart Streamlit.

---

## How the pipeline works

The pipeline lives in `radar/`, with **one file per NLP step**. Each step takes a
pandas DataFrame and adds new columns to it.

```
demo_corpus.csv
      │
      ▼
 collect.py ──► clean.py ──► langid.py ──► sentiment.py ──► radar.csv
 load posts     normalise    which         positive /
                the text     language?     negative / neutral
                                                  │
                                                  └──► embeddings.py   (only with --teaser)
                                                       meaning as numbers ──► embeddings.npy
```

### 1. `collect.py`: Data collection
Loads the corpus from CSV and checks that all required columns are present.
In a real project this would be a web scraper. The file explains the rules for
scraping responsibly (robots.txt, rate limits, no personal data).

### 2. `clean.py`: Text cleaning / normalisation
Makes a **lowercase copy** of each post without URLs, @mentions, repeated punctuation
(`!!!` → `!`) or extra spaces. The original text is never overwritten.
**Emoji and diacritics are kept on purpose**, because they carry sentiment and language clues.

### 3. `langid.py`: Language identification
Compares three detectors:

| Detector        | How it works                                                                   |
|-----------------|--------------------------------------------------------------------------------|
| `detect_naive`  | `langdetect`, a popular library. It knows Croatian but **not** Bosnian or Serbian. |
| `detect_lingua` | `lingua`, restricted to the 4 languages we expect (bs, hr, sr, en)              |
| `detect_ours`   | **Our own rules**: counts marker words that differ between the languages (*tko / ko*, *kruh / hljeb / hleb*, *mlijeko / mleko*, *je li / da li*), and detects code-switching by counting English and BCS words |

The result of `detect_ours` is stored in the `lang` column.

### 4. `sentiment.py`: Sentiment analysis
Compares two very different approaches:

| Model   | Type                         | Strengths / weaknesses                                                  |
|---------|------------------------------|-------------------------------------------------------------------------|
| VADER   | Rule-based word list         | Fast and transparent, but knows **English only**                        |
| XLM-R   | Transformer (machine learning) | Reads many languages, but is slower and a "black box"                  |

Sentiment reads the **original** text, not the cleaned one, because capitals
(`GREAT`) and `!!!` make an opinion stronger.

### 5. `embeddings.py`: Sentence embeddings (teaser, only with `--teaser`)
A sneak peek at later labs. Turns every post into a **vector of 384 numbers** with a multilingual MiniLM model. The
model was trained so that sentences with the same meaning get nearby vectors, **in any
of 50+ languages**: "the internet is down" and "LipaNet opet ne radi" end up close together.
How close two posts are is measured with **cosine similarity** (1.0 = same meaning).

With these vectors, the pipeline does three things:

| Function               | What it does                                                                 |
|------------------------|------------------------------------------------------------------------------|
| `semantic_search`      | Finds the posts closest in meaning to a query, whatever language they are in |
| `cluster_posts`        | Groups posts by meaning with **k-means**, without any labels (unsupervised learning). The result is stored in the `cluster` column |
| `near_duplicate_pairs` | Finds pairs of posts that say (almost) the same thing                         |

`map_2d` squeezes the 384 numbers down to 2 (with PCA), so every post can be drawn as a
dot on a map (`map_x`, `map_y` columns). k-means and PCA are written out by hand with
numpy, so you can read every step.

---

## The demo data

`data/demo/demo_corpus.csv` contains **300 synthetic posts** from four fictional
platforms (a forum, a chat, news comments and a confessions page).

| Language          | Posts |
|-------------------|-------|
| Bosnian (`bs`)    | 120   |
| English (`en`)    | 60    |
| Croatian (`hr`)   | 45    |
| Serbian (`sr`), Latin script | 45 |
| Mixed (`mixed`), code-switching | 30 |

275 of the posts are genuinely different; only 25 are variants of another post (typed
again without diacritics or with a typo). The data is **hard on purpose**: about 40% of
BCS posts have no diacritics, and there are slang words, typos, sarcasm, negation, very
short posts and code-switching.

Three **gold-label** columns hold the correct answers, decided by a human:

- `lang_gold`: the true language,
- `sentiment_gold`: the true sentiment,
- `topic_gold`: what the post is about (`exams`, `professors`, `wifi`, `events`,
  `cafeteria`, `enrollment`, `parking`, or `other`). Used by the teaser to check the k-means clusters.

Three more columns help with fair evaluation:

- `ambiguous`: `True` for 10 posts that are genuinely undecidable between Bosnian and
  Croatian (their gold label is `bs`, but `hr` is defensible),
- `sentiment_debatable`: `True` for 18 posts whose sentiment label is disputable
  (sarcasm, understatement),
- `split`: `dev` or `test` (see [Honest evaluation](#honest-evaluation-dev-and-test)).

They are used **only for evaluation**. The pipeline never reads them, because a real
system would not have the answers. The `lang` column is empty in the CSV, and the pipeline fills it.

The CSV is generated by `scripts/make_demo_data.py` (fixed seed, no APIs).
**Do not edit the CSV by hand.** To change the data, edit the script and re-run it:

```bash
uv run python scripts/make_demo_data.py
```

Full details are in [`data/demo/README.md`](data/demo/README.md).

---

## Lab 1: what you can change

Lines you are meant to experiment with are marked with:

```python
# LAB 1: change me
```

| File                 | Setting                          | What it controls                                             |
|----------------------|----------------------------------|--------------------------------------------------------------|
| `radar/langid.py`    | `MARKER_WORDS`                   | Words that point to Bosnian, Croatian or Serbian             |
| `radar/langid.py`    | `MIN_MARKER_LEAD`                | How much evidence a language needs to win                    |
| `radar/langid.py`    | `ENGLISH_WORDS`, `BCS_WORDS`     | Words used to spot English, BCS, and mixed posts             |
| `radar/langid.py`    | `MIXED_MIN_WORDS`                | How many words of each language make a post "mixed"         |
| `radar/sentiment.py` | `POSITIVE_THRESHOLD`, `NEGATIVE_THRESHOLD` | Where VADER's score turns into positive / neutral / negative |
| `radar/embeddings.py` | `NUMBER_OF_CLUSTERS`            | *Teaser:* how many groups k-means makes                       |
| `radar/embeddings.py` | `DUPLICATE_THRESHOLD`           | *Teaser:* how similar two posts must be to count as near-duplicates |

The workflow is always the same: **look at dev mistakes → edit → re-run the pipeline →
compare the test numbers**. Find new marker words in **dev** posts only; never copy words
from test posts, or your test score stops meaning anything.
The lab activities are in [`labs/lab01/`](labs/lab01/): a code tour (`tour.md`), a
break-and-fix task (`break_and_fix.md`) and an error analysis (`errors.md`).

### Adding your own dashboard tab

Every file in `radar/modules/` becomes a tab automatically. You never need to edit `app.py`.

The easiest start is the template: **copy `radar/modules/_template.py`** to a new name
without the leading `_`, e.g. `radar/modules/m07_my_tab.py`, then restart the dashboard.
The template lists every column in `df` and shows a number, a chart and a table.

Every tab file needs just two things:

```python
import streamlit as st

NAME = "My tab"          # the tab label


def render(df):          # df = the pipeline output, one row per post
    st.write(f"The corpus has {len(df)} posts.")
```

- Files are shown in alphabetical order, which is why they start with `m01_`, `m02_`, ...
- Files starting with `_` are skipped.
- If your tab crashes, only that tab shows the error. The other tabs keep working.

---

## Tests

```bash
uv run pytest            # fast tests, takes a few seconds
uv run pytest -m slow    # slow tests: need the downloaded XLM-R model
```

| File                      | What it checks                                                        |
|---------------------------|-----------------------------------------------------------------------|
| `tests/test_data.py`      | The demo corpus has the right columns, unique ids, no empty texts, valid `lang_gold` |
| `tests/test_clean.py`     | Cleaning removes URLs and @mentions, keeps diacritics and emoji, collapses "??!!" |
| `tests/test_langid.py`    | `detect_ours` gets 6 clear examples right                              |
| `tests/test_sentiment.py` | VADER on 3 clear English sentences; XLM-R (slow)                       |
| `tests/test_modules.py`   | **The contract for dashboard tabs:** every file in `radar/modules/` has `NAME` and `render(df)` |

Your own dashboard tabs must pass `tests/test_modules.py`. GitHub runs the fast tests on
every push (`.github/workflows/tests.yml`).

---

## Project structure

```
nlp-radar/
├── app.py                    # Streamlit dashboard (auto-loads radar/modules/)
├── check_setup.py            # Lab-day safety net: is everything installed?
├── radar/                    # The NLP pipeline
│   ├── collect.py            #   1. data collection
│   ├── clean.py              #   2. text cleaning
│   ├── langid.py             #   3. language identification
│   ├── sentiment.py          #   4. sentiment analysis
│   ├── embeddings.py         #   5. teaser: sentence embeddings (--teaser only)
│   └── modules/              # Dashboard tabs, one file per tab (_template.py = copy me)
├── scripts/
│   ├── download_models.py    # Download the model(s) once (needs internet)
│   ├── run_pipeline.py       # Run all stages and print the evaluation report
│   └── make_demo_data.py     # Generate the synthetic demo corpus
├── data/
│   ├── demo/                 # Synthetic input corpus (+ its README)
│   └── processed/            # Pipeline output (created by run_pipeline.py, not in git)
├── models/                   # Downloaded models (not in git)
├── labs/lab01/               # Lab 1 instructions
├── tests/                    # pytest tests (uv run pytest)
├── .github/workflows/        # Runs the tests on every push
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Exact package versions, so everyone gets the same setup
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Not sure what is wrong | Run `uv run python check_setup.py`: every ❌ line names its fix. |
| `uv: command not found` | Close and reopen the terminal after installing uv. |
| `Sentiment model not found` | Run `uv run python scripts/download_models.py` (needs internet once). |
| `Embedding model not found` | Only needed for `--teaser`: run `uv run python scripts/download_models.py --teaser`. |
| *Search by meaning* tab says it is a teaser | That tab needs `uv run python scripts/run_pipeline.py --teaser`. |
| Dashboard says *"No pipeline output yet"* | Run `uv run python scripts/run_pipeline.py` first. |
| Dashboard shows old results | Re-run the pipeline, then press **R** in the browser. Still old? Restart Streamlit (`Ctrl+C`, run it again). |
| One dashboard tab shows a red error box | That module has a bug. The error message names the file and line. |
| The pipeline is slow the first time | The models need a few seconds to load. Later runs are faster. |
| Out of memory | XLM-R needs about 1 GB of RAM (1.5 GB with `--teaser`). Close other programs (browser tabs!). |
| `ModuleNotFoundError` | You probably ran `python ...` instead of `uv run python ...`. |

---

## Credits

- Sentiment model: [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- Embedding model: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Libraries: [langdetect](https://pypi.org/project/langdetect/), [lingua](https://github.com/pemistahl/lingua-py),
  [VADER](https://github.com/cjhutto/vaderSentiment), [🤗 Transformers](https://huggingface.co/docs/transformers),
  [Streamlit](https://streamlit.io/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Plotly](https://plotly.com/python/)
