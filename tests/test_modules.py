"""The contract for dashboard tabs: every file in radar/modules/ must pass this.

app.py turns each file in radar/modules/ into a tab. It needs two things from
each file: NAME (the tab label, a string) and render(df) (draws the tab).
Your own weekly modules must pass these tests too.
"""

import importlib
from pathlib import Path

import pytest

MODULES_DIR = Path(__file__).resolve().parent.parent / "radar" / "modules"
# The same files app.py loads: files starting with "_" are skipped.
TAB_FILES = sorted(MODULES_DIR.glob("[!_]*.py"))


def load(path):
    return importlib.import_module(f"radar.modules.{path.stem}")


@pytest.mark.parametrize("path", TAB_FILES, ids=lambda path: path.name)
def test_has_name(path):
    """The tab file defines NAME, a non-empty string."""
    module = load(path)
    assert isinstance(getattr(module, "NAME", None), str), f"{path.name} needs NAME = \"...\""
    assert module.NAME.strip() != ""


@pytest.mark.parametrize("path", TAB_FILES, ids=lambda path: path.name)
def test_has_render(path):
    """The tab file defines a function render(df)."""
    module = load(path)
    assert callable(getattr(module, "render", None)), f"{path.name} needs def render(df):"
