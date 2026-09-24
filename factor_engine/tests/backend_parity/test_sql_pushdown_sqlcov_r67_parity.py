# -*- coding: utf-8 -*-
"""R67-SQLCOV: 真实 DuckDB 下推等价校验（Wilder-ATR 族 / Keltner / VWAP / Ichimoku 派生）。

每个算子都必须在**真实 DuckDB** 上全量下推执行（``assert_duckdb_real_sql_execution``：
``fully_sql``/``sql_fully_pushed``/``sql_fallback_subtree_count==0``），
并与 pandas backend 结果在 ``rtol=1e-12, atol=1e-12`` 内逐点一致（NaN 语义一致）。

数据源：pandas 侧用 ``InMemorySeriesSource``，SQL 侧用 parquet 落盘的
``data_access`` dataset —— 两条路径读的是同一份宽表面板。

共享工作树污染说明
------------------
同一树内既有模块 ``factor_engine/tests/test_ts_batch1_rank_if_direct.py`` 在**收集期**
（模块作用域）把伪造的 ``sys.modules['factor_engine.cleaned_operators']`` 及其
``.base`` 子模块替换为 stub，且因为它自身 import 抛 ValueError 而未执行还原逻辑，
于是同一 pytest 会话中**其后**收集的所有模块都会拿到假包（本树 70 个既有 module
级 ERROR 同源）。本文件对该污染显式处理：能修复则修复并正常执行，不能修复则
**明确 skip**（而非报红），并在健康树上完整执行。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

_REGISTRY_PKG = "factor_engine.cleaned_operators"

# --- 收集期污染守卫：若 canonical 名已被绑定到无 __file__ 的 stub，直接 skip ---
_bound = _sys.modules.get(_REGISTRY_PKG)
if _bound is not None and getattr(_bound, "__file__", None) is None:
    pytest.skip(
        "shared tree: sys.modules['factor_engine.cleaned_operators'] was replaced by a fake "
        "stub (test_ts_batch1_rank_if_direct.py leaks it and never restores it because its own "
        "import raises); registry is unimportable in this pytest session",
        allow_module_level=True,
    )

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

# 记录本模块导入时看到的**真实** registry 模块对象（含已加载的子模块），
# 供 fixture 在被污染时还原。
_REGISTRY_SNAPSHOT: dict[str, object] = {
    k: v
    for k, v in _sys.modules.items()
    if k == _REGISTRY_PKG or k.startswith(_REGISTRY_PKG + ".")
}

# canonical -> (callable name, tuple of trailing scalar args, pandas columns, duckdb columns)
_SPECS: dict[str, tuple[str, tuple, tuple[str, ...], tuple[str, ...]]] = {
    # ---- batch 1: Wilder-ATR family (dimensionless derivatives) ----
    "atr_pct": ("atr_pct", (14,), ("high", "low", "close"), ("High", "Low", "Close")),
    "atr_acceleration": ("atr_acceleration", (14,), ("high", "low", "close"), ("High", "Low", "Close")),
    "atr_zscore": ("atr_zscore", (14, 20), ("high", "low", "close"), ("High", "Low", "Close")),
    "atr_percentile": ("atr_percentile", (14, 20), ("high", "low", "close"), ("High", "Low", "Close")),
    "atr_short_long_ratio": ("atr_short_long_ratio", (7, 21), ("high", "low", "close"), ("High", "Low", "Close")),
    # ---- batch 2: Keltner family (dimensionless width / compression / breakout) ----
    "keltner_width_pct": ("keltner_width_pct", (20, 14, 2.0), ("high", "low", "close"), ("High", "Low", "Close")),
    "keltner_compression": ("keltner_compression", (20, 14, 2.0, 20), ("high", "low", "close"), ("High", "Low", "Close")),
    "keltner_breakout_strength": ("keltner_breakout_strength", (20, 14, 2.0), ("high", "low", "close"), ("High", "Low", "Close")),
    # ---- batch 2: rolling-VWAP strict-cohort family ----
    "vwap_distance_pct": ("vwap_distance_pct", (20,), ("close", "volume"), ("Close", "Volume")),
    "vwap_slope_pct": ("vwap_slope_pct", (20,), ("close", "volume"), ("Close", "Volume")),
    "vwap_premium_pct": ("vwap_premium_pct", (20,), ("high", "low", "close", "volume"), ("High", "Low", "Close", "Volume")),
    # ---- batch 3: causal Ichimoku-derived distances / cross ----
    "chikou_distance_pct": ("chikou_distance_pct", (26,), ("high", "low", "close"), ("High", "Low", "Close")),
    "senkou_span_causal_pct": ("senkou_span_causal_pct", (9, 26), ("high", "low", "close"), ("High", "Low", "Close")),
    "tenkan_kijun_cross": ("tenkan_kijun_cross", (9, 26), ("high", "low", "close"), ("High", "Low", "Close")),
}

_PANDAS_CANONICALS = sorted(_SPECS)


# ===========================================================================
# 共享树污染：还原被替换的 registry 模块（能还原则正常执行，否则 skip）
# ===========================================================================
def _restore_registry_modules() -> None:
    for key, mod in _REGISTRY_SNAPSHOT.items():
        if _sys.modules.get(key) is not mod:
            _sys.modules[key] = mod  # type: ignore[assignment]
    # 丢弃快照之外的、被塞进来的 stub 子模块，让后续 import 拿到真实实现
    for key in [
        k
        for k in list(_sys.modules)
        if k.startswith(_REGISTRY_PKG + ".")
        and k not in _REGISTRY_SNAPSHOT
        and getattr(_sys.modules[k], "__file__", None) is None
    ]:
        del _sys.modules[key]


def _registry_is_healthy() -> bool:
    mod = _sys.modules.get(_REGISTRY_PKG)
    base = _sys.modules.get(_REGISTRY_PKG + ".base")
    return (
        mod is not None
        and getattr(mod, "__file__", None) is not None
        and hasattr(mod, "load_all")
        and (base is None or hasattr(base, "validate_operator_call_arity"))
    )


# ===========================================================================
# 合成宽表面板（正常 / 含 NaN）
# ===========================================================================
def _make_panel(seed: int, *, n: int = 120, with_nan: bool = False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = ["A", "B", "C"]
    close = pd.DataFrame(
        {c: 60.0 + np.cumsum(rng.normal(0.0, 1.3, n)) + 15.0 * j for j, c in enumerate(cols)},
        index=idx,
    )
    spread = pd.DataFrame(
        {c: np.abs(rng.normal(1.2, 0.5, n)) + 0.3 for c in cols}, index=idx
    )
    high = close + spread
    low = close - spread
    vol = pd.DataFrame(
        {c: 1.0e6 + np.abs(rng.normal(0.0, 2.0e5, n)) for c in cols}, index=idx
    )
    if with_nan:
        for c, positions in (("A", [4, 5, 30, 61, 90]), ("B", [11, 40]), ("C", [70])):
            close.loc[idx[positions], c] = np.nan
        high.loc[idx[[17]], "A"] = np.nan
        low.loc[idx[[18]], "B"] = np.nan
        vol.loc[idx[[22]], "C"] = np.nan
    return close, high, low, vol


CASES = {
    "normal": lambda: _make_panel(11),
    "nan": lambda: _make_panel(12, with_nan=True),
}

_DATA = {"close": "close", "high": "high", "low": "low", "volume": "volume"}
_DUCK = {"close": "Close", "high": "High", "low": "Low", "volume": "Volume"}


# ===========================================================================
# 独立 numpy 参考（仅 atr_pct：不 import 被测模块，验证 ATR 基座口径）
# ===========================================================================
def _ref_tr(high, low, close):
    prev = np.concatenate([[np.nan], close[:-1]])
    return np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])


def _ref_wilder(vals: np.ndarray, w: int) -> np.ndarray:
    """pandas ``x.ewm(alpha=1/w, adjust=False, min_periods=w).mean()`` 复刻。

    NaN 行 carry-forward；有效观测数 < w 时输出 NaN。
    """
    alpha = 1.0 / float(w)
    out = np.full(vals.shape, np.nan)
    y = np.nan
    started = False
    cnt = 0
    for i, v in enumerate(vals):
        if np.isnan(v):
            out[i] = y if cnt >= w else np.nan
            continue
        y = v if not started else (1.0 - alpha) * y + alpha * v
        started = True
        cnt += 1
        out[i] = y if cnt >= w else np.nan
    return out


def _ref_atr_pct(close_df, high_df, low_df, w):
    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    for name in close_df.columns:
        c = close_df[name].to_numpy(dtype=float)
        h = high_df[name].to_numpy(dtype=float)
        low = low_df[name].to_numpy(dtype=float)
        atr = _ref_wilder(_ref_tr(h, low, c), w)
        # pandas ``close.where(close > 0)`` -> non-positive / NaN close gives NaN
        safe = np.where(c > 0.0, c, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = atr / safe
    return out


# ===========================================================================
# 引擎运行脚手架
# ===========================================================================
def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _series(run_out) -> pd.Series:
    result = run_out["result"]
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    return result.sort_index()


def _engine_expr(canonical, names, scalar_args):
    fn, _args, in_cols, _duck_cols = _SPECS[canonical]
    factory = make_cleaned_call_factory(fn)
    return factory(*[col(names[c]) for c in in_cols], *scalar_args)


def _to_long_source(close, high, low, vol) -> InMemorySeriesSource:
    idx = pd.MultiIndex.from_product(
        [close.index, close.columns], names=["timestamp", "instrument"]
    )
    data = {
        name: pd.Series(df.to_numpy(dtype=float).reshape(-1), index=idx)
        for name, df in (("close", close), ("high", high), ("low", low), ("volume", vol))
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
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, close, high, low, vol) -> None:
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
                    "Volume": float(vol.loc[ts, sym]),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _duckdb_source_for(tmp_path, monkeypatch, close, high, low, vol):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb(tmp_path / "data", close, high, low, vol)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


# ===========================================================================
# 1) 注册 / 三后端
# ===========================================================================
@pytest.fixture(scope="module", autouse=True)
def _loaded():
    _restore_registry_modules()
    if not _registry_is_healthy():
        pytest.skip(
            "shared tree: registry module clobbered by an unrelated test module "
            "(test_ts_batch1_rank_if_direct.py) and could not be restored"
        )
    load_all()


@pytest.mark.parametrize("canonical", _PANDAS_CANONICALS)
def test_registered_with_three_backends(canonical):
    entry = OperatorRegistry.catalog().get(canonical)
    assert entry is not None, f"{canonical} not in catalog"
    assert set(entry["backends"]) >= {"pandas_numpy", "polars", "sql"}, entry["backends"]


# ===========================================================================
# 2) pandas backend vs 独立 numpy 参考（ATR 基座口径）
# ===========================================================================
@pytest.mark.parametrize("case", ["normal"])
def test_atr_pct_matches_independent_reference(case):
    close, high, low, vol = CASES[case]()
    source = _to_long_source(close, high, low, vol)
    expr = _engine_expr("atr_pct", _DATA, (14,))
    got = _series(_run(source, expr, "pandas")).unstack()
    got = pd.DataFrame(
        got.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float),
        index=close.index,
        columns=close.columns,
    )
    ref = _ref_atr_pct(close, high, low, 14)
    assert np.array_equal(np.isnan(got.to_numpy()), np.isnan(ref.to_numpy()))
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-10, atol=1e-10)


# ===========================================================================
# 3) 真实 DuckDB 全量下推：pandas / polars / sql 三后端一致
# ===========================================================================
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("canonical", _PANDAS_CANONICALS)
def test_pandas_polars_sql_agree(case, canonical, tmp_path, monkeypatch):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    close, high, low, vol = CASES[case]()
    mem_source = _to_long_source(close, high, low, vol)
    duckdb_source = _duckdb_source_for(tmp_path, monkeypatch, close, high, low, vol)

    _fn, scalar_args, _in_cols, _duck_cols = _SPECS[canonical]
    pd_out = _series(_run(mem_source, _engine_expr(canonical, _DATA, scalar_args), "pandas"))
    pl_out = _series(_run(mem_source, _engine_expr(canonical, _DATA, scalar_args), "polars"))
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=1e-12, atol=1e-12)

    sql_run = _run(duckdb_source, _engine_expr(canonical, _DUCK, scalar_args), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _series(sql_run)

    assert np.array_equal(pd_out.isna().to_numpy(), sql_out.isna().to_numpy()), "NaN pattern mismatch"
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-12, atol=1e-12)
