"""Stage 1 - Data collection.

NLP step: getting the raw text. Every NLP project starts here, and the data you
collect limits everything that comes after: a model can only learn (and we can
only measure) what is in the corpus.

In this lab we load a ready-made SYNTHETIC corpus from a CSV file, so the lab
works offline and contains no real people's posts.

-------------------------------------------------------------------------------
In real life this stage is a SCRAPER.
-------------------------------------------------------------------------------
A real "radar" would download posts from websites, for example with:
  - `requests` + `BeautifulSoup` for simple HTML pages, or
  - `Playwright` for pages that need a real browser (JavaScript, scrolling).

Scraping is powerful and easy to misuse. If you ever write a scraper, follow
these rules:
  1. Respect robots.txt. It tells you which pages the site owner allows bots
     to visit. If it says "Disallow", you don't scrape that page.
  2. Be slow: at most 1 request per second. A fast scraper can overload a small
     website - for its owner, that looks like an attack.
  3. Never store usernames or personal data. Keep only what the analysis needs
     (text, date, source). Personal data is protected by law (e.g. GDPR), and
     you don't need to know WHO wrote a post to measure sentiment.
  4. Check the site's terms of service, and prefer an official API if one exists.

There is deliberately no live scraping code in this repo.
"""

from pathlib import Path

import pandas as pd

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_corpus.csv"

# Every later stage and the dashboard rely on these columns.
REQUIRED_COLUMNS = ["id", "source", "url", "date", "title", "text", "lang"]


def load_corpus(path=DEFAULT_CORPUS_PATH):
    """NLP step: data collection - load the corpus into a DataFrame.

    Why it matters: if a column is missing, every later stage breaks with a
    confusing error. Checking once here gives one clear error message instead.
    """
    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Corpus {path} is missing columns: {missing}")

    # Empty cells are read as NaN (a float). Use "" so text functions always get a string.
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    return df
