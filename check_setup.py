"""Check that your computer is ready for the lab.

Run it from the project folder with:
    uv run python check_setup.py

Every check prints one line: a tick if it is fine, a cross if not. Every
cross comes with the exact command that fixes it. Fix the problems from top to
bottom: a later check often fails only because an earlier one did.

Run it right after setup, while you still have internet, so you can still
download anything that is missing.
"""

import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models" / "xlmr-sentiment"
CORPUS_PATH = REPO_ROOT / "data" / "demo" / "demo_corpus.csv"
RADAR_PATH = REPO_ROOT / "data" / "processed" / "radar.csv"

EXPECTED_PYTHON = (3, 12)
PACKAGES = [
    "pandas", "streamlit", "vaderSentiment", "transformers", "torch",
    "langdetect", "lingua", "plotly",
]
CORPUS_COLUMNS = [
    "id", "source", "url", "date", "title", "text", "lang",
    "lang_gold", "sentiment_gold",
]
# Files that scripts/download_models.py saves for a complete model.
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json"]
TEST_SENTENCE = "Danas je predavanje bilo odlično!"


class CheckFailed(Exception):
    """Raised by a check that failed. The message is the command that fixes it."""


# --- The checks --------------------------------------------------------------
# Each check returns a short "what we found" text, or raises CheckFailed with
# the fix. Any other error is caught by run_check, so nothing can crash.

def check_python():
    version = sys.version_info
    found = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) != EXPECTED_PYTHON:
        raise CheckFailed(f"found Python {found}. Fix: uv sync, then always use: uv run python ...")
    return f"Python {found}"


def check_uv_environment():
    # uv puts the project's Python in .venv/ inside the project folder.
    expected = (REPO_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise CheckFailed(f"running {sys.executable}. Fix: uv sync, then: uv run python check_setup.py")
    return "running inside the project's .venv"


def check_packages():
    missing = []
    for package in PACKAGES:
        try:
            importlib.import_module(package)
        except Exception:
            missing.append(package)
    if missing:
        raise CheckFailed(f"cannot import {', '.join(missing)}. Fix: uv sync")
    return f"all {len(PACKAGES)} packages import"


def check_model_files():
    missing = []
    for name in MODEL_FILES:
        if not (MODEL_DIR / name).exists():
            missing.append(name)
    if missing:
        raise CheckFailed(
            f"missing in models/xlmr-sentiment/: {', '.join(missing)}. "
            "Fix: uv run python scripts/download_models.py"
        )
    return "found in models/xlmr-sentiment/"


def check_corpus():
    import pandas as pd

    if not CORPUS_PATH.exists():
        raise CheckFailed("data/demo/demo_corpus.csv missing. Fix: uv run python scripts/make_demo_data.py")
    df = pd.read_csv(CORPUS_PATH)
    missing = [column for column in CORPUS_COLUMNS if column not in df.columns]
    if missing:
        raise CheckFailed(f"columns {missing} missing. Fix: uv run python scripts/make_demo_data.py")
    return f"{len(df)} posts, all columns present"


def check_pipeline_output():
    if not RADAR_PATH.exists():
        raise CheckFailed("data/processed/radar.csv missing. Fix: uv run python scripts/run_pipeline.py")
    return "data/processed/radar.csv found"


def check_git():
    if shutil.which("git") is None:
        raise CheckFailed("git is not installed. Fix: install it from https://git-scm.com/downloads")
    missing = []
    for setting in ["user.name", "user.email"]:
        result = subprocess.run(["git", "config", "--global", setting], capture_output=True, text=True)
        if not result.stdout.strip():
            missing.append(setting)
    if "user.name" in missing:
        raise CheckFailed('git user.name not set. Fix: git config --global user.name "Your Name"')
    if "user.email" in missing:
        raise CheckFailed('git user.email not set. Fix: git config --global user.email "you@example.com"')
    return "git installed, user.name and user.email set"


def check_xlmr_runs():
    if not (MODEL_DIR / "config.json").exists():
        raise CheckFailed("no model to load (see the model check above). Fix: uv run python scripts/download_models.py")
    # Make "import radar" work, whatever folder this script was started from.
    sys.path.insert(0, str(REPO_ROOT))
    from radar.sentiment import sentiment_xlmr

    start = time.perf_counter()
    labels, _ = sentiment_xlmr([TEST_SENTENCE])
    seconds = time.perf_counter() - start
    return f'XLM-R says "{labels[0]}" for "{TEST_SENTENCE}" ({seconds:.1f} s)'


CHECKS = [
    ("Python version", check_python),
    ("uv environment", check_uv_environment),
    ("Packages", check_packages),
    ("XLM-R model files", check_model_files),
    ("Demo corpus", check_corpus),
    ("Pipeline output", check_pipeline_output),
    ("Git", check_git),
    ("XLM-R works", check_xlmr_runs),
]


# --- Running and printing ----------------------------------------------------

def pick_symbols():
    """Ticks and crosses if the terminal can print them, otherwise [OK] / [X].

    Old Windows terminals use an encoding without emoji; printing one there
    would crash the script, which is the last thing a setup check should do.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        "✅❌".encode(sys.stdout.encoding or "ascii")
        return "✅", "❌"
    except Exception:
        return "[OK]", "[X]"


def run_check(function):
    """Run one check. Returns (passed, message). Never raises."""
    try:
        return True, function()
    except CheckFailed as problem:
        return False, str(problem)
    except Exception as error:
        # An unexpected error is reported like any other failed check.
        return False, f"unexpected error: {error!r}. Fix: uv sync, then run this check again"


def main():
    ok_symbol, fail_symbol = pick_symbols()
    print("Checking your setup for nlp-radar ...\n")

    problems = 0
    for name, function in CHECKS:
        passed, message = run_check(function)
        if passed:
            print(f"{ok_symbol} {name}: {message}")
        else:
            print(f"{fail_symbol} {name}: {message}")
            problems += 1

    print()
    if problems == 0:
        print("All good, run: uv run streamlit run app.py")
        return 0
    print(f"{problems} problems, fix them top to bottom")
    return 1


if __name__ == "__main__":
    sys.exit(main())
