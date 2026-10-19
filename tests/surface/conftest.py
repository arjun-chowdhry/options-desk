"""Shared fixtures for the surface tests: synthetic markets with a known SVI surface.

Building the American chain and cleaning it takes a few seconds (CRR trees for every quote),
so those fixtures are session-scoped and built once. Real network access is blocked for every
test (``tests/conftest.py``), which proves the surface code runs offline.
"""

from __future__ import annotations

import pytest

from optdesk.data.clean import clean_chain
from optdesk.surface.svi import fit_chain
from optdesk.surface.synthetic import synthetic_chain


@pytest.fixture(scope="session")
def exact_market():
    """European quotes with no spread noise and no rounding: mids are exact model prices."""
    return synthetic_chain(american=False, noise=0.0, tick=None)


@pytest.fixture(scope="session")
def exact_clean(exact_market):
    return clean_chain(exact_market.raw, exact_market.curve)


@pytest.fixture(scope="session")
def market():
    """The realistic default: American, noisy mids, penny-rounded quotes."""
    return synthetic_chain()


@pytest.fixture(scope="session")
def market_clean(market):
    return clean_chain(market.raw, market.curve, american=True)


@pytest.fixture(scope="session")
def market_fits(market_clean):
    return fit_chain(market_clean)
