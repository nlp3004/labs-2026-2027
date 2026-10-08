"""Colours shared by every dashboard tab. NOT a tab itself: app.py skips
files whose name starts with "_".

Sentiment always has the same colour, in every chart:
  negative = orange, neutral = grey, positive = blue.

Why orange and blue, not the usual red and green? About 1 in 12 men cannot
tell red from green (red-green colour blindness). Orange and blue stay easy
to tell apart for them too. These are from the Okabe-Ito palette, which was
designed for exactly this.
"""

SENTIMENT_COLORS = {
    "negative": "#E69F00",  # orange
    "neutral": "#999999",   # grey
    "positive": "#0072B2",  # blue
}

# Negative and positive on the outside, neutral in the middle.
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
