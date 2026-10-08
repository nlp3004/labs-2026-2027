"""Stage 4 - Sentiment analysis.

NLP step: deciding whether a text expresses a positive, negative or neutral
opinion. This is what turns a pile of posts into a "radar": are students
happy or angry about the wifi, the cafeteria, the exams?

We compare two very different approaches:

  a) sentiment_vader - a RULE-BASED tool: a hand-made list of English words
     with a score each ("good" = +1.9, "terrible" = -2.5), plus rules for 
     "not", "very", "!!!" and CAPITALS. Fast, transparent, English only.
  b) sentiment_xlmr  - a MACHINE-LEARNING model (a transformer) that learned
     sentiment from thousands of labelled tweets in 8 languages. Slower and a
     black box, but it can read languages and phrasings nobody wrote rules for.

Both read the ORIGINAL text, not clean_text: capitals ("GREAT") and repeated
punctuation ("!!!") make a post stronger, and VADER uses exactly those signals.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Loading from a local folder takes a second; a progress bar would only
# clutter the pipeline's printed results.
transformers_logging.disable_progress_bar()

# =============================================================================
# Settings. Students: edit these and re-run the pipeline!
# =============================================================================

# VADER gives every text a "compound" score from -1 (very negative) to +1
# (very positive). These thresholds turn that number into a label:
#   score >= POSITIVE_THRESHOLD -> "positive"
#   score <= NEGATIVE_THRESHOLD -> "negative"
#   anything in between         -> "neutral"
# +-0.05 is the value recommended by VADER's authors.
POSITIVE_THRESHOLD = 0.05  # LAB 1: change me
NEGATIVE_THRESHOLD = -0.05  # LAB 1: change me

# =============================================================================

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "xlmr-sentiment"

# The model reads posts in groups ("batches"). 32 posts at a time is much
# faster than one by one, and still fits easily in a laptop's memory.
BATCH_SIZE = 32

# Building the VADER analyzer reads its word list from disk. We do it ONCE
# when this file is imported, and reuse it for every post.
VADER = SentimentIntensityAnalyzer()

# Loading the transformer takes several seconds and ~1 GB of memory, so we
# load it only when it is first needed, and then keep it here for later calls.
_xlmr_tokenizer = None
_xlmr_model = None


def score_to_label(score):
    """Turn a score from -1 to +1 into positive / negative / neutral."""
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def sentiment_vader(text):
    """NLP step: sentiment analysis with a rule-based lexicon (VADER).

    Why it matters: VADER is a list of ~7,500 ENGLISH words and emoticons, each
    with a hand-assigned score, plus rules for negation ("not good") and
    intensity ("very good!!!"). It adds up the scores of the words it knows.

    Why it struggles with Bosnian: "odlično" or "katastrofa" are simply not in
    its list. Unknown words count as 0, so a strongly negative BCS post often
    comes out as "neutral". It sometimes gets BCS posts right anyway - when
    they contain emoji (😍 😡), which VADER translates to English words, or
    English words like "top" and "super" that BCS speakers also use.

    Returns (label, compound score from -1 to +1).
    """
    score = VADER.polarity_scores(text)["compound"]
    return score_to_label(score), score


def load_xlmr():
    """Load the XLM-R tokenizer and model from models/ (only the first time)."""
    global _xlmr_tokenizer, _xlmr_model

    if _xlmr_model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Sentiment model not found in {MODEL_DIR}.\n"
                "Run: uv run python scripts/download_models.py"
            )
        _xlmr_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _xlmr_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        # eval() switches off training-only behaviour (e.g. dropout), so the
        # same post always gets the same prediction.
        _xlmr_model.eval()

    return _xlmr_tokenizer, _xlmr_model


def predict_batch(texts, tokenizer, model):
    """Run the model on one batch of texts. Returns (labels, confidences)."""
    # The tokenizer splits text into sub-word pieces and turns them into IDs.
    # padding: make all posts in the batch the same length.
    # truncation: cut very long posts to the model's maximum length.
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # no_grad: we only predict, we don't train, so PyTorch can skip the
    # bookkeeping needed for learning. Faster and uses less memory.
    with torch.no_grad():
        logits = model(**inputs).logits

    # softmax turns the 3 raw scores into probabilities that add up to 1.
    probabilities = torch.softmax(logits, dim=-1)
    confidences, label_ids = probabilities.max(dim=-1)

    labels = []
    for label_id in label_ids.tolist():
        labels.append(model.config.id2label[label_id].lower())
    return labels, confidences.tolist()


def sentiment_xlmr(texts):
    """NLP step: sentiment analysis with a multilingual transformer (XLM-R).

    Why it matters: XLM-RoBERTa was first PRETRAINED on 2.5 TB of web text in
    100 languages, learning how words relate by filling in missing words.
    It was then FINE-TUNED: trained a little more on tweets labelled
    positive/negative/neutral in 8 languages (Arabic, English, French, German,
    Hindi, Italian, Portuguese, Spanish). Pretraining gives it general
    language knowledge; fine-tuning teaches it this one task.

    Bosnian, Croatian and Serbian were NOT among the 8 fine-tuning languages,
    but they were in pretraining - so the model has to transfer what it
    learned about sentiment to languages it never saw labels for.

    Runs on the CPU, BATCH_SIZE posts at a time.
    Returns (list of labels, list of confidences from 0 to 1).
    """
    tokenizer, model = load_xlmr()
    texts = list(texts)

    all_labels = []
    all_confidences = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        labels, confidences = predict_batch(batch, tokenizer, model)
        all_labels.extend(labels)
        all_confidences.extend(confidences)

    return all_labels, all_confidences
