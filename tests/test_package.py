"""Placeholder test: proves the package installs and imports. Real tests arrive with M1."""

import importlib

import optdesk


def test_version_is_set():
    assert optdesk.__version__ == "0.1.0"


def test_subpackages_import():
    for name in ["pricing", "data", "surface", "sim", "app"]:
        importlib.import_module(f"optdesk.{name}")
