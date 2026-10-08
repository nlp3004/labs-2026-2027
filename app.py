"""NLP Radar dashboard. Run it with:  uv run streamlit run app.py

Every file in radar/modules/ becomes one tab. To add a tab, drop a new file
there that defines NAME (the tab label) and render(df) (draws the tab).
You never need to edit this file.
"""

import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "processed" / "radar.csv"
MODULES_DIR = APP_DIR / "radar" / "modules"

st.set_page_config(page_title="NLP Radar", layout="wide")
st.title("NLP Radar – Univerzitet Lipa")
st.warning("Demo uses SYNTHETIC data about a fictional university.")


# Streamlit re-runs this whole file on every click. cache_data remembers the
# result, so the CSV is read from disk only once - until the file changes.
# The file's "last modified" time is an argument, so a new pipeline run gives
# a new argument, and cache_data reads the new file. Just refresh the browser.
@st.cache_data
def load_data(last_modified):
    return pd.read_csv(DATA_PATH)


if not DATA_PATH.exists():
    st.info("No pipeline output yet. Run: uv run python scripts/run_pipeline.py")
    st.stop()
df = load_data(DATA_PATH.stat().st_mtime)

# Find the tab files. sorted() gives a fixed order, which is why the files
# start with a number: m01_..., m02_... Files starting with "_" are skipped.
module_files = sorted(MODULES_DIR.glob("[!_]*.py"))
tab_modules = []
for path in module_files:
    try:
        # "radar/modules/m01_overview.py" -> import radar.modules.m01_overview
        tab_modules.append(importlib.import_module(f"radar.modules.{path.stem}"))
    except Exception as error:
        # A broken file still gets a tab (named after the file) showing its error.
        tab_modules.append(error)

tab_names = [getattr(module, "NAME", path.stem) for module, path in zip(tab_modules, module_files)]
for tab, module in zip(st.tabs(tab_names), tab_modules):
    with tab:
        # One broken tab must not take down the others: catch its error here.
        try:
            if isinstance(module, Exception):
                raise module
            module.render(df)
        except Exception as error:
            st.exception(error)
