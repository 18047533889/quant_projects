"""
QE-P0-1: Pearson IC DAG must be fully separated from the Rank IC DAG.

Regression guard for the DAG cross-wiring bug: ``ic.pearson.std`` and
``ic.pearson.ir`` previously resolved to ``pearson_ic_series`` (the daily
Pearson series), not to the std / IR of that series.  The two DAGs must be
completely separate:

    ic.rank.daily    -> rank_ic_series   -> {mean, median, std, ir, hac_t, hac_p}
    ic.pearson.daily -> pearson_ic_series -> {mean, median, std, ir, hac_t, hac_p}

Every alias must resolve to a DISTINCT registry spec, and the pearson-family
std/ir aliases must resolve to pearson std/ir kernels (not the daily series).
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.registry.metrics import (
    CANONICAL_METRIC_ALIASES,
    get_metric,
    list_metrics,
    resolve_alias,
)

RANK_ALIASES = [
    "ic.rank.daily", "ic.rank.mean", "ic.rank.median", "ic.rank.std",
    "ic.rank.ir", "ic.rank.hac_t", "ic.rank.hac_p",
]
PEARSON_ALIASES = [
    "ic.pearson.daily", "ic.pearson.mean", "ic.pearson.std", "ic.pearson.ir",
]


def test_pearson_std_ir_do_not_resolve_to_daily_series():
    """ic.pearson.std / ic.pearson.ir must NOT resolve to pearson_ic_series."""
    for alias in ("ic.pearson.std", "ic.pearson.ir"):
        resolved = resolve_alias(alias)
        assert resolved != "pearson_ic_series", (
            f"{alias} must not resolve to the daily series; got {resolved!r}"
        )


def test_pearson_std_ir_resolve_to_distinct_specs():
    """pearson std/ir are distinct registry entries, not the daily series."""
    std_name = resolve_alias("ic.pearson.std")
    ir_name = resolve_alias("ic.pearson.ir")
    daily_name = resolve_alias("ic.pearson.daily")
    assert std_name != daily_name
    assert ir_name != daily_name
    assert std_name != ir_name
    # Each is a real registered spec.
    assert std_name in list_metrics()
    assert ir_name in list_metrics()


def test_pearson_std_ir_declare_pearson_method():
    """The pearson-family std/ir specs must declare ic_method='pearson'."""
    for alias in ("ic.pearson.std", "ic.pearson.ir"):
        spec = get_metric(resolve_alias(alias))
        assert spec.ic_method == "pearson", alias


def test_pearson_std_ir_consume_ic_series_artifact():
    """pearson std/ir are derived metrics over the daily Pearson series."""
    for alias in ("ic.pearson.std", "ic.pearson.ir"):
        spec = get_metric(resolve_alias(alias))
        assert "ICSeriesArtifact" in (spec.requires or []), alias


def test_rank_and_pearson_std_ir_are_distinct_specs():
    """rank.std/ir and pearson.std/ir must be four distinct registry entries."""
    rank_std = resolve_alias("ic.rank.std")
    rank_ir = resolve_alias("ic.rank.ir")
    pearson_std = resolve_alias("ic.pearson.std")
    pearson_ir = resolve_alias("ic.pearson.ir")
    names = {rank_std, rank_ir, pearson_std, pearson_ir}
    assert len(names) == 4, f"expected 4 distinct specs, got {names}"
