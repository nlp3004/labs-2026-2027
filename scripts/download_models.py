"""Download the models ONCE, so the lab runs offline afterwards.

Run it with:
    uv run python scripts/download_models.py             # what the lab needs
    uv run python scripts/download_models.py --teaser    # + the embedding teaser

This is the only script in the repo that needs the internet. It downloads
models from the Hugging Face Hub and saves each one to its own folder in models/:

  - "cardiffnlp/twitter-xlm-roberta-base-sentiment" (about 1 GB)
      -> models/xlmr-sentiment/      used by radar/sentiment.py
  - with --teaser also "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    (about 470 MB) -> models/multilingual-minilm/ used by radar/embeddings.py

After that, the code loads the models from those folders with no network calls.
A model that is already downloaded is skipped, so re-running this is quick.

The model folders are in .gitignore: they are too big for git, so every
student runs this script on their own computer.
"""

import argparse
from pathlib import Path

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_MODEL_DIR = MODELS_DIR / "xlmr-sentiment"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_DIR = MODELS_DIR / "multilingual-minilm"


def folder_size_mb(folder):
    """Add up the size of every file in a folder, in megabytes."""
    total_bytes = 0
    for path in folder.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    return total_bytes / (1024 * 1024)


def download(model_name, model_dir, model_class):
    """Download one model + its tokenizer and save both to model_dir."""
    # config.json is written last-ish by save_pretrained; if it is there, the
    # model was saved before, and we do not need to download it again.
    if (model_dir / "config.json").exists():
        print(f"Already downloaded: {model_name} ({model_dir.name}/), skipping.")
        return

    print(f"Downloading {model_name} (this can take a few minutes) ...")

    # A model always comes in two parts:
    #   - the tokenizer turns text into numbers (token IDs) the model can read,
    #   - the model turns those numbers into its output.
    # They must match, so we download and save both.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)

    # save_pretrained writes everything into ONE plain folder that
    # from_pretrained can load later without the internet.
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    print(f"Saved to {model_dir}")
    print(f"Size on disk: {folder_size_mb(model_dir):.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download the models for nlp-radar.")
    parser.add_argument(
        "--teaser", action="store_true",
        help="also download the embedding model for run_pipeline.py --teaser",
    )
    arguments = parser.parse_args()

    # The sentiment model ends in a CLASSIFIER (3 outputs: neg / neu / pos),
    # so it needs the "ForSequenceClassification" class.
    download(SENTIMENT_MODEL_NAME, SENTIMENT_MODEL_DIR, AutoModelForSequenceClassification)

    if arguments.teaser:
        # The embedding model has NO classifier: we only want its vectors, so
        # the plain AutoModel class is enough.
        download(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIR, AutoModel)
    else:
        print("Skipping the embedding model (only needed for the --teaser run).")


if __name__ == "__main__":
    main()
