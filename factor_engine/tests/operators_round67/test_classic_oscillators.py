# -*- coding: utf-8 -*-
"""R67 经典随机指标：KDJ / Stochastic %K-%D / Williams %R。

三层校验（缺一不可）：
  1. 独立 numpy/pandas 参考实现比对（不使用被测代码）；
  2. 覆盖 正常数据 / 含 NaN / 分母为 0（连续一字板）三类输入；
  3. pandas_numpy / polars / sql 三后端一致性 + DuckDB 真实下推。

参考口径（与被测实现独立书写）：
  RSV = 100*(close-LLV(low,N))/(HHV(high,N)-LLV(low,N))，滚动极值 min_periods=1；
  分母为 0 → RSV = 100；close 为 NaN 的行 → NaN。
  K = 2/3*K_prev + 1/3*RSV（首值 RSV），D 同理作用于 K，J = 3K-2D；
  %K = RSV，%D = SMA(%K, m)；%R = RSV(N) - 100。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

_ALPHA = 1.0 / 3.0
_DECAY = 1.0 - _ALPHA

CANONICALS = ["kdj_k", "kdj_d", "kdj_j", "stoch_k", "stoch_d", "williams_r"]


# ===========================================================================
# 独立 numpy 参考实现（不 import 被测模块）
# ===========================================================================
def _roll_extreme(vals: np.ndarray, window: int, take_max: bool) -> np.ndarray:
    """rolling min/max，``min_periods=1``：忽略 NaN，窗口内全 NaN 时返回 NaN。"""
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = np.max(valid) if take_max else np.min(valid)
    return out


def _ref_rsv(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    llv = _roll_extreme(low, window, take_max=False)
    hhv = _roll_extreme(high, window, take_max=True)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(close[i]) or np.isnan(hhv[i]) or np.isnan(llv[i]):
            continue
        if hhv[i] == llv[i]:
            out[i] = 100.0
        else:
            out[i] = 100.0 * (close[i] - llv[i]) / (hhv[i] - llv[i])
    return out


def _ref_ewm(vals: np.ndarray) -> np.ndarray:
    """pandas ``ewm(alpha=1/3, adjust=False, min_periods=1)`` 的独立复刻。

    绝对位置衰减 + 有效观测重归一化；NaN 行 carry-forward（未播种前为 NaN）。
    """
    n = len(vals)
    out = np.full(n, np.nan)
    y = np.nan
    last: int | None = None
    for i in range(n):
        if np.isnan(vals[i]):
            out[i] = y
            continue
        if last is None:
            y = vals[i]
        else:
            gap = i - last
            y = (_DECAY**gap * y + _ALPHA * vals[i]) / (_DECAY**gap + _ALPHA)
        last = i
        out[i] = y
    return out


def _ref_sma(vals: np.ndarray, window: int) -> np.ndarray:
    """rolling mean，``min_periods=1``：忽略 NaN。"""
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = float(np.mean(valid))
    return out


def _ref_kdj(close, high, low, window):
    rsv = _ref_rsv(close, high, low, window)
    k = _ref_ewm(rsv)
    d = _ref_ewm(k)
    j = 3.0 * k - 2.0 * d
    return k, d, j


def _ref_stoch_d(close, high, low, window, smooth):
    return _ref_sma(_ref_rsv(close, high, low, window), smooth)


def _ref_williams_r(close, high, low, window):
    return _ref_rsv(close, high, low, window) - 100.0


def _ref_reference(canonical, close_df, high_df, low_df, window, smooth=3):
    """按 canonical 逐列生成独立参考结果（返回同形 DataFrame）。"""
    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    for name in close_df.columns:
        c = close_df[name].to_numpy(dtype=float)
        h = high_df[name].to_numpy(dtype=float)
        low = low_df[name].to_numpy(dtype=float)
        if canonical in ("kdj_k", "kdj_d", "kdj_j"):
            k, d, j = _ref_kdj(c, h, low, window)
            series = {"kdj_k": k, "kdj_d": d, "kdj_j": j}[canonical]
        elif canonical == "stoch_k":
            series = _ref_rsv(c, h, low, window)
        elif canonical == "stoch_d":
            series = _ref_stoch_d(c, h, low, window, smooth)
        elif canonical == "williams_r":
            series = _ref_williams_r(c, h, low, window)
        else:  # pragma: no cover
            raise AssertionError(canonical)
        out[name] = series
    return out


def _registry_calc(canonical, close_df, high_df, low_df, window, smooth=3):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"missing pandas_numpy backend for {canonical}"
    if canonical == "stoch_d":
        return op.calculate(close_df, high_df, low_df, window, smooth)
    return op.calculate(close_df, high_df, low_df, window)


def _max_abs_err(a: pd.DataFrame, b: pd.DataFrame) -> float:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    both_nan = np.isnan(av) & np.isnan(bv)
    assert np.array_equal(np.isnan(av), np.isnan(bv)), "NaN pattern mismatch"
    if both_nan.all():
        return 0.0
    return float(np.max(np.abs(av[~both_nan] - bv[~both_nan])))


# ===========================================================================
# 合成数据：正常 / 含 NaN / 连续一字板（分母为 0）
# ===========================================================================
def _make_wide(seed: int, *, n: int = 40, with_nan: bool = False, flat_run=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A", "B"]
    close = pd.DataFrame(
        {c: 50.0 + np.cumsum(rng.normal(0.0, 1.0, n)) + 10.0 * j for j, c in enumerate(cols)},
        index=idx,
    )
    spread = pd.DataFrame(
        {c: np.abs(rng.normal(1.0, 0.4, n)) + 0.2 for c in cols}, index=idx
    )
    high = close + spread
    low = close - spread
    if flat_run is not None:
        start, length = flat_run
        for c in cols:
            price = float(close[c].iloc[start])
            sl = slice(start, start + length)
            close.loc[idx[sl], c] = price
            high.loc[idx[sl], c] = price
            low.loc[idx[sl], c] = price
    if with_nan:
        for c, positions in (("A", [3, 4, 12, 25]), ("B", [5, 18])):
            close.loc[idx[positions], c] = np.nan
        high.loc[idx[[6]], "A"] = np.nan
        low.loc[idx[[7]], "A"] = np.nan
    return close, high, low


CASES = {
    "normal": lambda: _make_wide(1),
    "nan": lambda: _make_wide(2, with_nan=True),
    # 16 连续一字板（bar 8..23）：两个窗口（9 / 14）都能整窗落入，分母为 0
    "flat_zero_range": lambda: _make_wide(3, flat_run=(8, 16)),
}


# ===========================================================================
# 1) 注册 / DSL 解析
# ===========================================================================
@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


def test_registered_with_three_backends():
    for canonical in CANONICALS:
        entry = OperatorRegistry.catalog().get(canonical)
        assert entry is not None, f"{canonical} not in catalog"
        assert set(entry["backends"]) >= {"pandas_numpy", "polars", "sql"}, entry["backends"]


def test_daily_surface_migration_recorded():
    import factor_engine.cleaned_operators.operator_surface as surface

    for canonical in CANONICALS:
        record = surface.REVIEWED_MIGRATION_MANIFEST.get(canonical)
        assert record is not None and record["review_id"] == f"R67-{canonical}"
        assert surface.daily_factor_migrated() >= {canonical}
        assert canonical in surface.extended_only_canonicals()


@pytest.mark.parametrize(
    "expr",
    [
        "kdj_k(close, high, low, 9)",
        "kdj_d(close, high, low, 9)",
        "kdj_j(close, high, low, 9)",
        "stoch_k(close, high, low, 9)",
        "stoch_d(close, high, low, 9, 3)",
        "williams_r(close, high, low, 14)",
        "kdj(close, high, low, 9)",
        "willr(close, high, low, 14)",
    ],
)
def test_dsl_parse(expr):
    from factor_engine.api.dsl_parser import parse_expr

    parse_expr(expr, surface="all")
    parse_expr(expr, surface="daily")


# ===========================================================================
# 2) 独立参考实现比对（正常 / NaN / 分母为 0）
# ===========================================================================
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("canonical", CANONICALS)
def test_matches_independent_reference(case, canonical):
    close, high, low = CASES[case]()
    window = 14 if canonical == "williams_r" else 9
    got = _registry_calc(canonical, close, high, low, window)
    ref = _ref_reference(canonical, close, high, low, window)
    assert _max_abs_err(got, ref) < 1e-10


def test_zero_range_run_pins_rsv_to_100():
    close, high, low = CASES["flat_zero_range"]()
    k = _registry_calc("kdj_k", close, high, low, 9)
    stoch_k = _registry_calc("stoch_k", close, high, low, 9)
    williams = _registry_calc("williams_r", close, high, low, 14)
    # 一字板 bar 8..23；bar 16..23 的 9 窗口、bar 21..23 的 14 窗口完全落入其中
    flat9 = close.index[16:24]
    flat14 = close.index[21:24]
    assert np.array_equal(high.loc[flat14].to_numpy(), low.loc[flat14].to_numpy())
    assert np.array_equal(high.loc[flat9].to_numpy(), low.loc[flat9].to_numpy())
    # 分母为 0 → RSV = 100；%R = RSV - 100 = 0
    assert np.allclose(stoch_k.loc[flat9].to_numpy(), 100.0)
    assert np.allclose(williams.loc[flat14].to_numpy(), 0.0)
    # K 是该 RSV 的 1/3 权重递推：一字板期间 K_t = 2/3*K_{t-1} + 1/3*100
    kv = k.loc[flat9].to_numpy(dtype=float)
    expected = (2.0 / 3.0) * kv[:-1] + (1.0 / 3.0) * 100.0
    assert np.allclose(kv[1:], expected, atol=1e-12)
    assert np.all(kv[1:] > kv[:-1])  # 单调收敛至 100
    # %K / %R 值域
    for name, frame, lo_bound, hi_bound in (
        ("stoch_k", stoch_k, 0.0, 100.0),
        ("williams_r", williams, -100.0, 0.0),
    ):
        arr = frame.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        assert finite.min() >= lo_bound - 1e-9, name
        assert finite.max() <= hi_bound + 1e-9, name


def test_nan_close_rows_are_nan_for_rsv_family():
    close, high, low = CASES["nan"]()
    stoch_k = _registry_calc("stoch_k", close, high, low, 9)
    nan_positions = close.isna()
    assert stoch_k[nan_positions].isna().all().all()


# ===========================================================================
# 3) 三后端一致性（pandas / polars / sql）
# ===========================================================================
_EXPR_SIGNATURE = {
    "kdj_k": ("kdj_k", (9,)),
    "kdj_d": ("kdj_d", (9,)),
    "kdj_j": ("kdj_j", (9,)),
    "stoch_k": ("stoch_k", (9,)),
    "stoch_d": ("stoch_d", (9, 3)),
    "williams_r": ("williams_r", (14,)),
}


def _to_long_source(close, high, low) -> InMemorySeriesSource:
    idx = pd.MultiIndex.from_product(
        [close.index, close.columns], names=["timestamp", "instrument"]
    )
    data = {
        name: pd.Series(df.to_numpy(dtype=float).reshape(-1), index=idx)
        for name, df in (("close", close), ("high", high), ("low", low))
    }
    return InMemorySeriesSource(data=data)


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    High: double
    Low: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, close, high, low) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts in close.index:
        for sym in close.columns:
            rows.append(
                {
                    "TradeDate": ts.date(),
                    "Symbol": sym,
                    "Close": float(close.loc[ts, sym]),
                    "High": float(high.loc[ts, sym]),
                    "Low": float(low.loc[ts, sym]),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _duckdb_source_for(tmp_path, monkeypatch, close, high, low):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb(tmp_path / "data", close, high, low)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _series(run_out) -> pd.Series:
    result = run_out["result"]
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    return result.sort_index()


def _engine_expr(canonical, names):
    fn, args = _EXPR_SIGNATURE[canonical]
    factory = make_cleaned_call_factory(fn)
    return factory(col(names["close"]), col(names["high"]), col(names["low"]), *args)


_PANDAS_NAMES = {"close": "close", "high": "high", "low": "low"}
_DUCK_NAMES = {"close": "Close", "high": "High", "low": "Low"}


@pytest.mark.parametrize("case", list(CASES))
def test_pandas_polars_sql_agree(case, tmp_path, monkeypatch):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    close, high, low = CASES[case]()
    mem_source = _to_long_source(close, high, low)
    duckdb_source = _duckdb_source_for(tmp_path, monkeypatch, close, high, low)

    for canonical in CANONICALS:
        pd_out = _series(_run(mem_source, _engine_expr(canonical, _PANDAS_NAMES), "pandas"))
        pl_out = _series(_run(mem_source, _engine_expr(canonical, _PANDAS_NAMES), "polars"))
        pd.testing.assert_series_equal(
            pd_out, pl_out, check_names=False, rtol=1e-12, atol=1e-12
        )

        sql_run = _run(duckdb_source, _engine_expr(canonical, _DUCK_NAMES), "duckdb_sql")
        assert_duckdb_real_sql_execution(sql_run)
        sql_out = _series(sql_run)
        pd.testing.assert_series_equal(
            pd_out, sql_out, check_names=False, rtol=1e-11, atol=1e-11
        )

        # all three agree with the independent reference on the same long grid
        window = 14 if canonical == "williams_r" else 9
        ref = _ref_reference(canonical, close, high, low, window)
        ref_long = ref.stack()
        ref_long.index = ref_long.index.set_names(["timestamp", "instrument"])
        ref_long = ref_long.sort_index()
        pd.testing.assert_series_equal(
            pd_out.reindex(ref_long.index),
            ref_long,
            check_names=False,
            rtol=1e-10,
            atol=1e-10,
        )
