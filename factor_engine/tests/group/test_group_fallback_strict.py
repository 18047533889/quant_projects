# -*- coding: utf-8 -*-
"""R24-088..099 + R24-246: group operators declare their behavioral policies in
the contract, reject unknown policy strings, and never silently reindex the
group panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded

ensure_cleaned_loaded()
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _fixture(rows=4, cols=3):
    idx = pd.date_range("2024-01-01", periods=rows)
    names = ["A", "B", "C"][:cols]
    x = pd.DataFrame(np.arange(1.0, rows * cols + 1.0).reshape(rows, cols), index=idx, columns=names)
    g = pd.DataFrame([["X", "X", "Y"]] * rows, index=idx, columns=names)
    return x, g


@pytest.mark.parametrize("canon", ["group_demean", "group_mean", "group_sum", "group_rank_weighted_value"])
def test_unknown_fallback_policy_raises(canon) -> None:
    # R24-246 golden: fallback_policy="nna" must raise, never degrade to global.
    x, g = _fixture()
    op = OperatorRegistry.get(canon)
    with pytest.raises(ValueError, match="fallback_policy"):
        op.calculate(x, g, fallback_policy="nna")


@pytest.mark.parametrize("canon", ["group_demean", "group_mean", "group_sum", "group_rank_weighted_value"])
def test_declared_policy_in_contract(canon) -> None:
    # R24-094/219: the behavioral kwarg is a declared, searchable=False param.
    x, g = _fixture()
    op = OperatorRegistry.get(canon)
    spec = op.metadata.param_specs.get("fallback_policy")
    assert spec is not None, canon
    assert spec.searchable is False
    assert "fallback_policy" in op.metadata.param_names, canon


@pytest.mark.parametrize("canon", ["group_demean", "group_mean", "group_sum", "group_rank_weighted_value"])
def test_shuffled_group_columns_raise(canon) -> None:
    # R24-097: a shuffled group panel is a semantic corruption — never silently
    # reindexed.
    x, g = _fixture()
    op = OperatorRegistry.get(canon)
    with pytest.raises(ValueError, match="columns"):
        op.calculate(x, g[["C", "A", "B"]])


def test_group_demean_no_group_is_not_global() -> None:
    # R24-088..090: missing group must NOT silently become global demean.
    x, g = _fixture()
    op = OperatorRegistry.get("group_demean")
    out = op.calculate(x, g, fallback_policy="nan")
    # With groups present, no problem; the fail-closed path is group=None.
    # group=None + default nan → NaN (not a global demean).
    out_none = op.calculate(x, None)
    assert out_none.isna().all().all()
