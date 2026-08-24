# -*- coding: utf-8 -*-
"""R23 fundamental PIT certification test session (one shared load_all)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def r23_load_all() -> None:
    """One load_all per pytest session for all R23 fundamental tests."""
    from factor_engine.cleaned_operators import load_all

    load_all()
