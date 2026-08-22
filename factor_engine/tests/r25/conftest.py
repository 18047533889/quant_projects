# -*- coding: utf-8 -*-
"""R25 genuine-usability / evidence-independence test session."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def r25_load_all() -> None:
    from cleaned_operators import load_all

    load_all()
