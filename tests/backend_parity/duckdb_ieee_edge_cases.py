# -*- coding: utf-8
"""DuckDB IEEE NaN/Inf edge case 注册（原生 fixture，不经 Pandas/Parquet）。"""
from __future__ import annotations

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col


def _duck_col(name: str):
    mapping = {
        "close": "Close",
        "open": "Open",
        "ret": "Ret",
        "numer": "Numer",
        "denom": "Denom",
        "neg": "Neg",
        "flag": "Flag",
        "grp": "Grp",
        "exp": "Exp",
        "volume": "Volume",
    }
    return col(mapping.get(name, name))


def _edge(name: str):
    """Build a NAN_REQUIRED-primitive expression on the DuckDB edge fixture.

    Mirrors ``tests/backend_parity/test_p0_semantic_edges._edge_expr`` so the
    case registry's declared edge dimensions match the genuinely verified ones.
    """
    f = make_cleaned_call_factory(name)
    c = _duck_col("close")
    if name == "cs_mean":
        return f(c)
    if name == "cs_std":
        return f(c)
    if name == "cs_sum":
        return f(c)
    if name == "normalize":
        return f(c)
    if name == "zscore":
        return f(c)
    if name == "winsorize":
        return f(c)
    if name == "group_mean":
        return f(c, _duck_col("grp"))
    if name == "group_std":
        return f(c, _duck_col("grp"))
    if name == "group_zscore":
        return f(c, _duck_col("grp"))
    if name == "group_normalize":
        return f(c, _duck_col("grp"))
    if name == "group_rank":
        return f(c, _duck_col("grp"))
    if name == "ts_mean":
        return f(c, 2)
    if name == "ts_std":
        return f(c, 2)
    if name == "ts_var":
        return f(c, 2)
    if name == "ts_zscore":
        return f(c, 2)
    if name == "ts_sharpe":
        return f(c, 2)
    if name == "ts_corr":
        return f(c, _duck_col("exp"), 2)
    if name == "ts_cov":
        return f(c, _duck_col("exp"), 2)
    if name == "ts_beta":
        return f(c, _duck_col("exp"), 2)
    if name == "maximum":
        return f(c, _duck_col("neg"))
    if name == "minimum":
        return f(c, _duck_col("neg"))
    if name == "where":
        return f(_duck_col("flag"), c, _duck_col("exp"))
    if name == "coalesce":
        return f(c, _duck_col("flag"))
    raise KeyError(name)


# Daily NAN_REQUIRED primitives (review P0-A02): every op whose declared edge
# contract includes NaN/Inf handling gets a genuinely verified edge case via
# tests/backend_parity/test_p0_semantic_edges.py (three-backend NaN parity +
# NaN/Inf != 0 coercion).  scale/volatility are non-daily, so
# ``sync_primitive_evidence`` filters them out of the daily edge lists.
_NAN_REQUIRED_DAILY = (
    "cs_mean", "cs_std", "cs_sum", "normalize", "zscore", "winsorize",
    "group_mean", "group_std", "group_zscore", "group_normalize", "group_rank",
    "ts_mean", "ts_std", "ts_var", "ts_zscore", "ts_sharpe",
    "ts_corr", "ts_cov", "ts_beta", "maximum", "minimum", "where", "coalesce",
)

DUCKDB_IEEE_NAN_CASES = [
    ("is_nan", lambda: make_cleaned_call_factory("is_nan")(_duck_col("close"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(_duck_col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(_duck_col("close"))),
    *[(n, lambda n=n: _edge(n)) for n in _NAN_REQUIRED_DAILY],
]

DUCKDB_IEEE_INF_CASES = [
    ("is_finite", lambda: make_cleaned_call_factory("is_finite")(_duck_col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(_duck_col("close"), 0.0)),
    *[(n, lambda n=n: _edge(n)) for n in _NAN_REQUIRED_DAILY],
]