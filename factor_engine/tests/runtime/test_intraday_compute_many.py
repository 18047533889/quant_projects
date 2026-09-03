# -*- coding: utf-8
"""P0#5: IntradayFeatureCompiler.compute_many single-scan equivalence tests.

``compute_many`` must compute N minute-feature operators from ONE scan of the
minute panel (one (day, bar, inst) grid materialization), and its results must
be byte-identical (rtol/atol 1e-12) to the standalone per-operator
``_calculate_series`` path — including on random panels with NaN gaps and
all-NaN days.

Grid-routed operators (PERF-2 whitelisted vector kernels) reuse the shared
grid; the scan counter proves the single-scan property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.intraday_aggregator import IntradayFeatureCompiler


def _random_panel(
    seed: int,
    *,
    n_inst: int = 3,
    n_days: int = 5,
    bars_per_day: int = 240,
    drop_pct: float = 0.08,
) -> dict[str, pd.Series]:
    """MultiIndex(timestamp, instrument) minute panel with NaN gaps + all-NaN day.

    - ``drop_pct``: close/volume 中随机置 NaN 的行占比（真缺失缺口）。
    - 第 0 天 S00000 全天无 bar（全 NaN 交易日），聚合必须产出 NaN。
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-03-04", periods=n_days, freq="D")
    rows: list[tuple[pd.Timestamp, str]] = []
    for d, day in enumerate(days):
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(minutes=b)
            for i in range(n_inst):
                if d == 0 and i == 0:
                    continue  # S00000 第 0 天无任何 bar
                rows.append((ts, f"S{i:02d}"))
    idx = pd.MultiIndex.from_tuples(rows, names=["timestamp", "instrument"])
    n = len(idx)
    close = pd.Series(rng.normal(10, 2, n), index=idx, dtype="float64")
    volume = pd.Series(rng.integers(100, 10000, n).astype(float), index=idx)

    drop_c = rng.choice(n, size=int(n * drop_pct), replace=False)
    drop_v = rng.choice(n, size=int(n * (drop_pct + 0.04)), replace=False)
    close.iloc[drop_c] = np.nan
    volume.iloc[drop_v] = np.nan
    return {"close": close, "volume": volume}


def _assert_panel_equal(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert got.index.equals(expected.index), (
        f"索引不等: got={got.index} exp={expected.index}"
    )
    assert got.columns.equals(expected.columns), (
        f"列不等: got={got.columns} exp={expected.columns}"
    )
    g = got.to_numpy(dtype="float64")
    e = expected.to_numpy(dtype="float64")
    gnan, enan = np.isnan(g), np.isnan(e)
    assert np.array_equal(gnan, enan), (
        f"NaN 位置不一致: {int((gnan != enan).sum())} 处不同"
    )
    np.testing.assert_allclose(
        g[~gnan], e[~enan], rtol=1e-12, atol=1e-12, equal_nan=True
    )


# --------------------------------------------------------------------- #
# 1. Single-scan equivalence vs per-op path
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_days", [3, 5])
def test_compute_many_equals_per_op(seed: int, n_days: int) -> None:
    panels = _random_panel(seed, n_days=n_days)
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])

    # Per-op standalone path (each operator re-scans the panel).
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    per_op: dict[str, pd.DataFrame] = {}
    for name in (
        "intra_session_mean_reversion",
        "intra_price_delay",
        "intra_volume_imbalance",
    ):
        op = OperatorRegistry.get(name, mode="any")
        assert op is not None, name
        args = []
        for p in op.metadata.param_names:
            args.append(IntradayFeatureCompiler._to_wide(panels[p]))
        per_op[name] = op.calculate(*args)

    # compute_many: ONE scan for all three.
    out = compiler.compute_many(
        [
            "intra_session_mean_reversion",
            "intra_price_delay",
            "intra_volume_imbalance",
        ],
        panels=panels,
    )

    assert set(out) == set(per_op)
    for name in per_op:
        _assert_panel_equal(out[name], per_op[name])


# --------------------------------------------------------------------- #
# 2. Scan counter: ONE scan for N operators
# --------------------------------------------------------------------- #

def test_compute_many_single_scan_for_three_operators() -> None:
    panels = _random_panel(7, n_days=4)
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])
    assert compiler.scan_count == 0

    out = compiler.compute_many(
        [
            "intra_session_mean_reversion",
            "intra_price_delay",
            "intra_volume_imbalance",
        ],
        panels=panels,
    )
    assert len(out) == 3
    # ONE grid materialization for three operators, not three.
    assert compiler.scan_count == 1, f"scan_count={compiler.scan_count}"


def test_scan_count_increments_per_distinct_field_set() -> None:
    panels = _random_panel(8, n_days=3)
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])
    compiler.compute_many(["intra_session_mean_reversion"], panels=panels)
    assert compiler.scan_count == 1
    # A second compute_many with the SAME field set reuses the cached grid.
    compiler.compute_many(["intra_session_mean_reversion"], panels=panels)
    assert compiler.scan_count == 1
    # A different field set (close+volume) needs a second grid materialization.
    compiler.compute_many(["intra_price_delay"], panels=panels)
    assert compiler.scan_count == 2


# --------------------------------------------------------------------- #
# 3. Empty panel edge case
# --------------------------------------------------------------------- #

def test_compute_many_empty_panel_raises() -> None:
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])
    with pytest.raises(ValueError):
        compiler.compute_many(["intra_session_mean_reversion"], panels={})


def test_compute_many_requires_panels_or_source() -> None:
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])
    with pytest.raises(ValueError, match="panels 或 source"):
        compiler.compute_many(["intra_session_mean_reversion"])


def test_compute_many_unknown_operator_raises() -> None:
    panels = _random_panel(9, n_days=3)
    compiler = IntradayFeatureCompiler(fields=["close", "volume"])
    with pytest.raises(ValueError, match="未知 intraday 算子"):
        compiler.compute_many(["not_a_real_operator"], panels=panels)
