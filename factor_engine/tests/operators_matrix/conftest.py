# -*- coding: utf-8 -*-
"""Full-operator matrix fixtures (runner lives in helpers.py)."""
from factor_engine.tests.operators_matrix.helpers import (  # noqa: F401
    BACKENDS,
    MANIFEST,
    OP_TIMEOUT,
    SyntheticPanelSource,
    _build_engine,
    _child_run,
    _checksum,
    _extract,
    _parity,
    _run_ops_forked,
)

import pytest


@pytest.fixture(scope="session")
def manifest():
    return MANIFEST


@pytest.fixture(scope="session")
def matrix(manifest):
    """{backend: {op: record}} for all parsable ops on all four backends."""
    ops = [n for n, v in MANIFEST.items() if v["expr"] and not v["skip"]]
    return {be: _run_ops_forked(be, ops) for be in BACKENDS}
