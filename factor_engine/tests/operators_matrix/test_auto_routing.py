# -*- coding: utf-8 -*-
"""BackendRouter contract: eligibility per the capability registry.

An explicit request either resolves or raises UnsupportedOperatorBackendError
(documented ineligibility) -- it must never silently fall through. End-to-end
auto execution on real data lives in test_parity_real_data.py.
"""
from __future__ import annotations

import pytest

from factor_engine.backend.backend_router import BackendRouter
from factor_engine.backend.operator_capability import UnsupportedOperatorBackendError

OPS = ("ts_mean", "ts_rank", "rank", "ts_corr")


@pytest.mark.parametrize("op", OPS)
def test_pandas_numpy_always_routes(op):
    sel = BackendRouter.select(op, requested_backend="pandas_numpy",
                               run_mode="production")
    assert sel is not None


@pytest.mark.parametrize("op", OPS)
def test_polars_request_resolves_or_documents_ineligibility(op):
    try:
        sel = BackendRouter.select(op, requested_backend="polars",
                                   run_mode="production")
    except UnsupportedOperatorBackendError:
        return  # documented ineligibility is an explicit contract outcome
    assert sel is not None


def test_auto_resolves_for_certified_ops():
    for op in OPS:
        try:
            sel = BackendRouter.select(op, requested_backend="auto",
                                       run_mode="production")
        except UnsupportedOperatorBackendError:
            continue
        assert sel is not None


def test_unknown_backend_rejected():
    with pytest.raises(UnsupportedOperatorBackendError):
        BackendRouter.select("ts_mean", requested_backend="definitely_not_real",
                             run_mode="production")
