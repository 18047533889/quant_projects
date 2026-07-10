# -*- coding: utf-8
"""Research fast path 全覆盖：SQL_IMPLEMENTED ∩ backend emitter。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_sql_implemented_subset_of_research_fastpath(_loaded):
    from backend.fastpath_allowlists import research_fastpath_allowlist
    from backend.production_fastpath_tiers import P2_RESEARCH_ONLY
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    research = research_fastpath_allowlist()
    skip = {"column", "literal"} | P2_RESEARCH_ONLY
    missing = sorted(c for c in SQL_IMPLEMENTED_CANONICALS if c not in skip and c not in research)
    assert not missing, f"SQL implemented 未进 research fastpath: {missing[:15]}"


def test_p1_extended_tier(_loaded):
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.production_fastpath_tiers import P1_EXTENDED_CANONICALS, P1_EXTENDED_CUM_PRODUCTION_SAFE

    python_rolling = {"WMA", "ts_skew", "ts_quantile", "Slope", "ts_argmax", "ts_argmin"}
    for canon in sorted(P1_EXTENDED_CANONICALS):
        tier = infer_polars_long_tier(canon)
        if canon in python_rolling or canon == "ts_decay_linear":
            assert tier == "python_rolling", canon
        elif canon in P1_EXTENDED_CUM_PRODUCTION_SAFE:
            assert tier == "native", canon
        else:
            assert tier in {"native", "python_rolling"}, canon
