# -*- coding: utf-8 -*-
"""R21-ROUTING-AUTHORITY test session.

Loads the operator registry once and registers SQL backends so the four
routing entry points can be exercised against the same capability state.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def r21_routing_load_all() -> None:
    from cleaned_operators import load_all

    load_all()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
