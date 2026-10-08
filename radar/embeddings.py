"""Stage 5 - Sentence embeddings (meaning as numbers).

NLP step: turning each post into a VECTOR, a list of 384 numbers, so that
posts with a similar MEANING get similar vectors, even when they share no words
and are written in different languages:

    "the internet is down"        -> [0.02, -0.11, 0.07, ...]
    "LipaNet opet ne radi"        -> [0.03, -0.09, 0.06, ...]   close!
    "menza danas super"           -> [-0.08, 0.12, -0.01, ...]  far away

The model (multilingual MiniLM) was trained on millions of sentence pairs that
mean the same thing, in 50+ languages, and learned to give them nearby vectors.

Once every post is a vector, three things become easy:
  a) semantic search      - find the posts closest in meaning to a query,
  b) clustering           - group posts by meaning, with NO labels (k-means),
  c) near-duplicate check - find posts that are almost the same.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

# Loading from a local folder takes a second; a progress bar would only
# clutter the pipeline's printed results.
transformers_logging.disable_progress_bar()

# =============================================================================
# Settings. Students: edit these and re-run the pipeline!
# =============================================================================

# How many groups k-means should split the posts into. The data has 8 gold
# topics, but k-means does not know that: try other numbers and compare.
NUMBER_OF_CLUSTERS = 8  # LAB 1: change me

# Two posts with a similarity at or above this count as near-duplicates.
# 1.0 = identical meaning. Lower = more pairs, but also more false alarms.
DUPLICATE_THRESHOLD = 0.90  # LAB 1: change me

# =============================================================================

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "multilingual-minilm"

# The model reads posts in groups ("batches"), like in radar/sentiment.py.
BATCH_SIZE = 32

# k-means starts from random posts, so different starts give different
# clusters. We try several starts and keep the best one. A fixed seed makes
# the result the same on every run and every computer.
KMEANS_SEED = 0
KMEANS_RESTARTS = 5
KMEANS_MAX_STEPS = 100

# Loaded the first time it is needed, then kept here (like XLM-R).
_tokenizer = None
_model = None


def load_embedder():
    """Load the embedding model and its tokenizer from models/ (only the first time)."""
    global _tokenizer, _model

    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding model not found in {MODEL_DIR}.\n"
                "Run: uv run python scripts/download_models.py"
            )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModel.from_pretrained(MODEL_DIR)
        _model.eval()  # prediction mode: no dropout, same input -> same output

    return _tokenizer, _model


def embed_batch(texts, tokenizer, model):
    """Turn one batch of texts into one vector per text."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        # The model gives one vector PER TOKEN: shape (texts, tokens, 384).
        token_vectors = model(**inputs).last_hidden_state

    # We want one vector per TEXT, so we average the token vectors ("mean
    # pooling"). The mask is 1 for real tokens and 0 for padding, so the
    # padding added to short texts does not count in the average.
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    text_vectors = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1)

    # Scale every vector to length 1. Then the similarity of two vectors is
    # simply their dot product (this is called cosine similarity).
    text_vectors = torch.nn.functional.normalize(text_vectors, dim=1)
    return text_vectors.numpy()


def embed_texts(texts):
    """NLP step: sentence embeddings - one meaning vector per text.

    Why it matters: computers cannot compare meanings, but they can compare
    numbers. Once texts are vectors, "similar meaning" becomes "nearby
    vectors", and we can search, group and de-duplicate posts in any language.

    Returns a numpy array with one row (384 numbers) per text.
    """
    tokenizer, model = load_embedder()
    texts = list(texts)

    batches = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        batches.append(embed_batch(batch, tokenizer, model))
    return np.vstack(batches)


def similarities(query_vector, vectors):
    """Cosine similarity of one vector to every row of vectors.

    1.0 = same meaning, around 0 = unrelated. Because all vectors have length 1,
    the cosine similarity is just the dot product, which numpy's "@" computes.
    """
    return vectors @ query_vector


def semantic_search(query, vectors, top_k):
    """NLP step: semantic search - the posts closest in MEANING to a query.

    Why it matters: keyword search finds "wifi" only in posts that contain
    "wifi". Semantic search also finds "LipaNet opet ne radi" and "net ne radi
    u domu" - different words, different language, same meaning.

    Near-duplicates of a result already found are skipped: ten copies of the
    same post with different emoji would only push other good results away.

    Returns (row numbers of the best posts, their similarity scores), best first.
    """
    query_vector = embed_texts([query])[0]
    scores = similarities(query_vector, vectors)

    best_rows = []
    # argsort sorts from low to high; we want the highest scores first.
    for row in np.argsort(-scores):
        if len(best_rows) == top_k:
            break
        is_duplicate = False
        for chosen in best_rows:
            if vectors[row] @ vectors[chosen] >= DUPLICATE_THRESHOLD:
                is_duplicate = True
        if not is_duplicate:
            best_rows.append(row)
    return best_rows, scores[best_rows]


def kmeans(vectors, k, rng):
    """Group vectors into k clusters. Returns (cluster number per vector, quality).

    The algorithm, step by step:
      1. Pick k random posts as the starting cluster centres.
      2. Put every post in the cluster whose centre is most similar to it.
      3. Move every centre to the average of the posts in its cluster.
      4. Repeat 2-3 until no post changes cluster.
    """
    start_rows = rng.choice(len(vectors), size=k, replace=False)
    centres = vectors[start_rows]
    clusters = np.full(len(vectors), -1)

    for _ in range(KMEANS_MAX_STEPS):
        # Step 2: similarity of every post to every centre -> pick the best.
        new_clusters = np.argmax(vectors @ centres.T, axis=1)
        if np.array_equal(new_clusters, clusters):
            break  # nothing changed: we are done
        clusters = new_clusters

        # Step 3: move each centre to the middle of its posts.
        for cluster in range(k):
            members = vectors[clusters == cluster]
            if len(members) > 0:  # an empty cluster keeps its old centre
                centre = members.mean(axis=0)
                centres[cluster] = centre / np.linalg.norm(centre)

    # Quality: how similar posts are, on average, to their own centre.
    quality = np.mean(np.sum(vectors * centres[clusters], axis=1))
    return clusters, quality


def cluster_posts(vectors, k=NUMBER_OF_CLUSTERS):
    """NLP step: clustering - group posts by meaning WITHOUT any labels.

    Why it matters: in real life nobody has labelled your posts with topics.
    Clustering discovers groups on its own ("unsupervised learning"). It does
    not NAME the groups: cluster 3 is just "3", and a human reads the posts
    to see what it is about.

    Returns one cluster number (0 to k-1) per post.
    """
    rng = np.random.default_rng(KMEANS_SEED)
    best_clusters = None
    best_quality = -1.0
    for _ in range(KMEANS_RESTARTS):
        clusters, quality = kmeans(vectors, k, rng)
        if quality > best_quality:
            best_clusters = clusters
            best_quality = quality
    return best_clusters


def near_duplicate_pairs(vectors, threshold=DUPLICATE_THRESHOLD):
    """NLP step: near-duplicate detection - pairs of posts that say the same thing.

    Why it matters: the same post copied with a typo or an emoji is still the
    same post. Duplicates make a corpus look bigger than it is, and if one
    copy is used to build a tool and another to test it, the test looks
    better than it should.

    Returns a list of (row a, row b, similarity), most similar first.
    """
    all_similarities = vectors @ vectors.T
    pairs = []
    for a in range(len(vectors)):
        # Start at a + 1: compare each pair once, and never a post with itself.
        for b in range(a + 1, len(vectors)):
            if all_similarities[a, b] >= threshold:
                pairs.append((a, b, float(all_similarities[a, b])))
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs


def map_2d(vectors):
    """Squeeze 384 numbers per post down to 2, so we can draw posts on a map.

    This is PCA (principal component analysis): it finds the 2 directions in
    which the vectors differ the most, and measures every post along them.
    Much detail is lost, but posts that are close in 384 dimensions usually
    stay close on the map.

    Returns an array with one (x, y) row per post.
    """
    centred = vectors - vectors.mean(axis=0)
    # SVD finds the main directions; the first rows of directions are the
    # directions with the most variation.
    _, _, directions = np.linalg.svd(centred, full_matrices=False)
    return centred @ directions[:2].T
