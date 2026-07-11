# -*- coding: utf-8
"""DuckDB IEEE NaN/Inf edge case 注册（原生 fixture，不经 Pandas/Parquet）。"""
from __future__ import annotations

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col


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
        "volume": "Volume",
    }
    return col(mapping.get(name, name))


DUCKDB_IEEE_NAN_CASES = [
    ("is_nan", lambda: make_cleaned_call_factory("is_nan")(_duck_col("close"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(_duck_col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(_duck_col("close"))),
]

DUCKDB_IEEE_INF_CASES = [
    ("is_finite", lambda: make_cleaned_call_factory("is_finite")(_duck_col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(_duck_col("close"), 0.0)),
]