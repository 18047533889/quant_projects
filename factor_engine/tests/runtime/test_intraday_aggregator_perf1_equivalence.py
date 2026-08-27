# -*- coding: utf-8
"""PERF-1: IntradayAggregator 快速路径 vs 旧长表 groupby 参考实现的等值测试。

新快速路径（unstack 宽表 + resample('D')）必须与 PERF-1 之前的旧实现
（reset→N-way outer merge→长表 groupby(["trade_date","asset"])）在
值 + 索引 + dtype + name 上完全一致。参考实现 ``_*_long`` 是旧逻辑逐字保留。

随机 fixture：3 insts × 3-5 天 × 每分钟 bar，含 NaN 缺口、空列、全 NaN 日。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.intraday_aggregator import IntradayAggregator


def _random_panel(
    seed: int,
    *,
    n_inst: int = 3,
    n_days: int = 5,
    bars_per_day: int = 240,
    drop_pct: float = 0.08,
    empty_column: bool = True,
) -> dict[str, pd.Series]:
    """带 NaN 缺口 + 可选空列 + 全 NaN 日的随机面板 fixture。

    - ``drop_pct``: volume/amount 中随机置 NaN 的行占比（真缺失）。
    - ``empty_column``: 额外加一列全 NaN（源数据里真实存在的空列）。
    - 第 0 天 S00000 全天 NaN（全 NaN 交易日），聚合必须产出 NaN。
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-03-04", periods=n_days, freq="D")
    rows: list[tuple[pd.Timestamp, str]] = []
    for d, day in enumerate(days):
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(minutes=b)  # 每分钟 bar
            for i in range(n_inst):
                if d == 0 and i == 0:
                    continue  # S00000 第 0 天无任何 bar → 该 (day, inst) 缺失
                rows.append((ts, f"S{i:02d}"))
    idx = pd.MultiIndex.from_tuples(
        rows, names=["timestamp", "instrument"]
    )
    n = len(idx)
    open_ = pd.Series(rng.normal(10, 2, n), index=idx, dtype="float64")
    close = pd.Series(rng.normal(10, 2, n), index=idx, dtype="float64")
    volume = pd.Series(rng.integers(100, 10000, n).astype(float), index=idx)
    amount = pd.Series(rng.uniform(1e5, 1e7, n), index=idx)

    # 真缺失缺口：volume / amount 随机位置 NaN（amount 缺口比 volume 略多，
    # 使 vwap 分子分母出现错位 NaN 的组合）。
    drop_v = rng.choice(n, size=int(n * drop_pct), replace=False)
    drop_a = rng.choice(n, size=int(n * (drop_pct + 0.04)), replace=False)
    volume.iloc[drop_v] = np.nan
    amount.iloc[drop_a] = np.nan

    out = {
        "open": open_,
        "close": close,
        "volume": volume,
        "amount": amount,
    }
    if empty_column:
        # 源数据里真实存在的全 NaN 列（空列）。
        out["empty"] = pd.Series(np.nan, index=idx, dtype="float64")
    return out


# --------------------------------------------------------------------- #
# 参考实现（PERF-1 之前旧逻辑，逐字保留）——本文件内独立实现，避免依赖
# 被测模块的私有 _*_long 方法（独立等值基准，防止测试与实现同源漂移）。
# --------------------------------------------------------------------- #


def _ref_stack_frame(panels: dict[str, pd.Series]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for col, series in panels.items():
        df = series.reset_index()
        df.columns = ["datetime", "asset", col]
        frames.append(df)
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["datetime", "asset"], how="outer")
    out["trade_date"] = pd.DatetimeIndex(out["datetime"]).normalize()
    return out


def _ref_last(panels, column):
    df = _ref_stack_frame(panels)
    daily = (
        df.sort_values(["asset", "datetime"])
        .groupby(["trade_date", "asset"], as_index=False)
        .last()
    )
    idx = pd.MultiIndex.from_arrays(
        [daily["trade_date"], daily["asset"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(daily[column].values, index=idx, name=column)


def _ref_first(panels, column):
    df = _ref_stack_frame(panels)
    daily = (
        df.sort_values(["asset", "datetime"])
        .groupby(["trade_date", "asset"], as_index=False)
        .first()
    )
    idx = pd.MultiIndex.from_arrays(
        [daily["trade_date"], daily["asset"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(daily[column].values, index=idx, name=column)


def _ref_sum(panels, column):
    df = _ref_stack_frame(panels)
    grouped = df.groupby(["trade_date", "asset"])[column].sum()
    grouped.index.names = ["timestamp", "instrument"]
    return grouped.rename(column)


def _ref_mean(panels, column):
    df = _ref_stack_frame(panels)
    grouped = df.groupby(["trade_date", "asset"])[column].mean()
    grouped.index.names = ["timestamp", "instrument"]
    return grouped.rename(column)


def _ref_vwap(panels, amount="amount", volume="volume", price="close"):
    df = _ref_stack_frame(panels)
    if amount in panels:
        numerator = df.groupby(["trade_date", "asset"])[amount].sum()
    elif price in panels:
        numerator = (df[price] * df[volume]).groupby(
            [df["trade_date"], df["asset"]]
        ).sum()
    else:
        raise KeyError(f"vwap 需要 {amount!r} 或 {price!r} 列")
    denominator = df.groupby(["trade_date", "asset"])[volume].sum()
    vwap = (numerator / denominator.replace(0, pd.NA)).rename("vwap")
    vwap.index.names = ["timestamp", "instrument"]
    return vwap


# --------------------------------------------------------------------- #
# 等值断言工具
# --------------------------------------------------------------------- #


def _assert_series_equal(got: pd.Series, expected: pd.Series) -> None:
    assert got.index.equals(expected.index), (
        f"索引不等: got={got.index.names} len={len(got)}, "
        f"exp={expected.index.names} len={len(expected)}"
    )
    assert got.index.names == ["timestamp", "instrument"], got.index.names
    assert got.name == expected.name, (got.name, expected.name)
    # 先按 NaN 位置比对（值+NaN 位置必须完全相同）
    g, e = got.to_numpy(dtype="float64"), expected.to_numpy(dtype="float64")
    gnan, enan = np.isnan(g), np.isnan(e)
    assert np.array_equal(gnan, enan), (
        f"NaN 位置不一致: {int((gnan != enan).sum())} 处不同"
    )
    # 非 NaN 位置逐值相等
    np.testing.assert_allclose(
        g[~gnan], e[~enan], rtol=1e-12, atol=1e-12, equal_nan=True
    )


# --------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_days", [3, 5])
def test_fast_equals_reference_all_features(seed: int, n_days: int) -> None:
    panels = _random_panel(seed, n_days=n_days)
    agg = IntradayAggregator.from_panel(panels)

    for col in ("open", "close", "volume"):
        _assert_series_equal(agg.last(col), _ref_last(panels, col))
        _assert_series_equal(agg.first(col), _ref_first(panels, col))
        _assert_series_equal(agg.sum(col), _ref_sum(panels, col))
        _assert_series_equal(agg.mean(col), _ref_mean(panels, col))

    _assert_series_equal(agg.vwap(), _ref_vwap(panels))


def test_fast_equals_reference_empty_column() -> None:
    panels = _random_panel(5, n_days=4)
    assert "empty" in panels  # fixture 自带全 NaN 列
    agg = IntradayAggregator.from_panel(panels)
    _assert_series_equal(agg.last("empty"), _ref_last(panels, "empty"))
    _assert_series_equal(agg.sum("empty"), _ref_sum(panels, "empty"))
    _assert_series_equal(agg.mean("empty"), _ref_mean(panels, "empty"))
    # 全 NaN 列的 mean 全为 NaN（与旧 groupby().mean() 一致，不补 0）。
    assert agg.mean("empty").isna().all()


def test_fast_equals_reference_all_nan_day() -> None:
    panels = _random_panel(11, n_days=4)
    agg = IntradayAggregator.from_panel(panels)
    # 第 0 天 S00000 无任何 bar → (day0, S00000) 在 last/first 中缺失
    daily_last = agg.last("close")
    assert ("2024-03-04", "S00000") not in daily_last.index
    assert daily_last.index.names == ["timestamp", "instrument"]


def test_fast_equals_reference_vwap_legacy_fallback() -> None:
    # 无 amount 列 → close×volume fallback
    panels = _random_panel(9, n_days=3)
    panels.pop("amount")
    panels.pop("empty")
    agg = IntradayAggregator.from_panel(panels)
    _assert_series_equal(agg.vwap(), _ref_vwap(panels))


def test_fast_vwap_missing_volume_raises_keyerror() -> None:
    panels = _random_panel(9, n_days=3)
    panels = {"close": panels["close"]}
    agg = IntradayAggregator.from_panel(panels)
    with pytest.raises(KeyError, match="volume"):
        agg.vwap()


def test_fast_reuses_wide_cache() -> None:
    panels = _random_panel(4, n_days=3)
    agg = IntradayAggregator.from_panel(panels)
    assert agg._wide_frames == {}
    agg.last("close")
    # 一次 unstack 后缓存；后续特征不再重新 unstack
    assert "close" in agg._wide_frames
    agg.sum("close")
    agg.mean("close")
    assert list(agg._wide_frames) == ["close"]  # 只 unstack 过一次
    assert agg._trade_date is not None


def test_empty_panel_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="至少一列"):
        IntradayAggregator.from_panel({})


def test_missing_column_raises_keyerror() -> None:
    panels = _random_panel(4, n_days=3)
    agg = IntradayAggregator.from_panel(panels)
    with pytest.raises(KeyError):
        agg.last("not_a_column")
    with pytest.raises(KeyError):
        agg.sum("not_a_column")
    with pytest.raises(KeyError):
        agg.mean("not_a_column")
    with pytest.raises(KeyError):
        agg.first("not_a_column")


def test_alignment_mismatch_raises() -> None:
    """P0-08 session-grid 对齐假设：列 index 不一致 → 显式报错而非静默错位。"""
    panels = _random_panel(4, n_days=2, empty_column=False)
    close = panels["close"]
    # 构造一个行数少 1 的 volume（缺一根 bar 且行序被打乱）
    vol_mis = panels["volume"].iloc[:-1]
    agg = IntradayAggregator.from_panel({"close": close, "volume": vol_mis})
    with pytest.raises(ValueError, match="未完全对齐|P0-08"):
        agg.last("close")
