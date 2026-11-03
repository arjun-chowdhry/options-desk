"""Placeholder test: proves the package installs and imports. Real tests arrive with M1."""

import importlib

import optdesk


def test_version_is_set():
    assert optdesk.__version__ == "0.1.0"


def test_subpackages_import():
    for name in ["pricing", "data", "surface", "sim", "app"]:
        importlib.import_module(f"optdesk.{name}")


def test_requirements_pin_every_dependency():
    """Streamlit Cloud installs from requirements.txt: it must cover pyproject's dependencies."""
    import re
    import tomllib
    from pathlib import Path

    repo = Path(__file__).parents[1]
    deps = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["dependencies"]
    lines = [
        line.strip()
        for line in (repo / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    # every requirement is an exact pin or has an upper bound: the server never installs an
    # untested major or minor version
    bounded = [line for line in lines if line != "-e ." and ("==" in line or "<" in line)]
    assert len(bounded) == len(lines) - 1, lines
    pinned = {re.split(r"[<>=!~\[ ]", line)[0].lower() for line in bounded}
    for dep in deps:
        assert re.split(r"[<>=!~\[ ]", dep)[0].lower() in pinned, dep
    assert "-e ." in lines  # the package itself
