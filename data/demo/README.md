# Demo corpus: SYNTHETIC teaching data, not real posts

`demo_corpus.csv` contains **300 synthetic social-media posts** about
**Univerzitet Lipa**, a **fictional** university in the **fictional** city of Brezovac.

- Every post was written by hand for IT 3004 Lab 1.
- All people (professors, students), platforms ("Forum Lipa", "Kampus Chat",
  "Novine Brezovac komentari", "Lipa Confessions"), apps ("eLipa", "LipaNet") and
  places are **invented**. Any resemblance to real people or institutions is accidental.
- URLs are placeholders (`https://example.com/post/<id>`).

The file is created by `scripts/make_demo_data.py`. That script uses a fixed random
seed and calls no APIs, so re-running it gives exactly the same CSV:

```
uv run python scripts/make_demo_data.py
```

## Columns

| Column                | Meaning                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `id`                  | Post number (1-300). Not in date order: sort by `date` for time.         |
| `source`              | Fictional platform the post comes from                                   |
| `url`                 | Placeholder link                                                         |
| `date`                | Posting date, between Sep 2025 and Sep 2026 (with three planted events, see below) |
| `title`               | Forum thread title or news headline; **empty** for chat and confessions  |
| `text`                | The post itself                                                          |
| `lang`                | **Empty on purpose**: the pipeline's language-identification step fills it in |
| `lang_gold`           | The true language, labelled by a human (see below)                       |
| `sentiment_gold`      | The true sentiment, labelled by a human: `positive`, `negative` or `neutral` |
| `topic_gold`          | What the post is about, labelled by a human (see below)                  |
| `ambiguous`           | `True` for the 10 posts that are genuinely undecidable between Bosnian and Croatian |
| `sentiment_debatable` | `True` for the 18 posts whose sentiment label reasonable people could dispute |
| `split`               | `dev` or `test`: which half of the data the post belongs to (see below)  |

"Gold" labels are the human-decided correct answers. We compare our tools' output
against them to measure how well the tools work. The pipeline **never reads** the gold
columns, the two flags or `split`: they are only used to evaluate the tools.

### `lang_gold` values

| Value   | Language                                           | Posts |
|---------|----------------------------------------------------|-------|
| `bs`    | Bosnian                                            | 120   |
| `hr`    | Croatian                                           | 45    |
| `sr`    | Serbian (Latin script)                             | 45    |
| `en`    | English                                            | 60    |
| `mixed` | Code-switching between BCS and English (e.g. "exam je bio insane, ne mogu vise") | 30 |

10 of the Bosnian posts are **`ambiguous`**: they use only words and forms that are
equally correct in Bosnian and Croatian (e.g. "Ispit je pomjeren za ponedjeljak u devet.").
Their gold label is `bs` because the writer is Bosnian, but a detector that answers `hr`
is not really wrong. Nobody could tell from the text alone.

### `sentiment_gold` values

110 negative, 95 positive, 95 neutral.

### `topic_gold` values

The lab pipeline does not use topics. The optional teaser (`run_pipeline.py --teaser`)
groups posts by meaning with k-means clustering and checks those clusters against `topic_gold`.

| Value        | About                                               | Posts |
|--------------|-----------------------------------------------------|-------|
| `exams`      | Exams, midterms (*kolokvij*), results               | 51    |
| `events`     | Lipa Fest, career days, concerts, parties           | 46    |
| `professors` | Lecturers, lectures, office hours                   | 42    |
| `cafeteria`  | The cafeteria (*menza*), food, coffee               | 39    |
| `wifi`       | LipaNet, internet in the library and dorms          | 39    |
| `enrollment` | Enrollment, the eLipa portal, the student office    | 35    |
| `parking`    | Parking on campus                                   | 27    |
| `other`      | Short posts and campus life (lifts, heating, lost backpacks) | 21 |

## Three planted events

Most posts get a random date in the year, but three events are hidden in the dates,
so the sentiment-over-time chart has something to find:

| Event                | When                    | What happens                                        |
|----------------------|-------------------------|-----------------------------------------------------|
| January exam period  | 12 Jan - 6 Feb 2026     | 3 in 4 negative exam posts are dated here            |
| Wifi outage          | 9 - 20 Mar 2026         | 3 in 4 negative wifi posts are dated here            |
| Lipa Fest            | 15 - 24 May 2026        | 3 in 4 positive event posts are dated here           |

## The dev / test split

Every post is in one of two halves, 50/50, **separately for every language** (so each
half has 60 Bosnian posts, 30 English posts, and so on):

- **`dev`**: look at these posts as much as you like, and tune your rules on them
  (e.g. new marker words for `detect_ours`).
- **`test`**: only for the final score. Do not read these posts while tuning.

Why? If you add a marker word because you saw it in a post, your detector will of
course get that post right. Measured on the same posts, it would look better than it
really is. Measured on posts it has never seen, the score tells you how it would do
on new posts. A post and its variant (see below) are always in the same half, so a
twin cannot leak the answer from dev into test.

## Why this data is hard for NLP tools (on purpose)

Real online text is messy. This corpus is designed so you will **see tools fail**:

- **No diacritics:** about 40% of Bosnian/Croatian/Serbian posts are typed without
  č ć š ž đ ("sta", "cetvrtak", "nista"), just as many people type on their phones.
- **Closely related languages:** Bosnian, Croatian and Serbian share most of their
  vocabulary, and a short post can be valid in all three. 10 posts are genuinely
  undecidable (`ambiguous`).
- **Code-switching:** posts that mix two languages in one sentence.
- **Slang, typos and emoji:** "jok", "haman", "merak", "jel", 🙃.
- **Sarcasm:** "super, opet nema wifi 🙃" uses positive words but is negative.
- **Negation:** "Nije loše predavanje danas" ("Not a bad lecture today") is *positive*.
- **Very short posts:** "ok", "jok", "ugh", "meh" give a tool almost nothing to work with.
- **Debatable sentiment:** 18 posts (e.g. "Ispit je bio... zanimljiv.",
  "The cafeteria changed the menu. Bold move.") are deliberately unclear. Their gold
  label is one annotator's best judgement, and you may well disagree.

## Different posts, few copies

275 of the 300 posts are **genuinely different** hand-written posts. The other 25 are
**variants**: the same post typed again without diacritics, with a typo or about a
different professor. Variants are kept rare on purpose, because near-identical copies
make a corpus look bigger than it is and make tools look better than they are. Only 19
pairs of posts are near-duplicates (similarity ≥ 0.90 in the embedding teaser). The first
50 rows contain no variants at all.
