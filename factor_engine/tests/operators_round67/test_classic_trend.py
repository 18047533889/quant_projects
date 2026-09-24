# -*- coding: utf-8 -*-
"""R67 经典趋势震荡指标：CCI / BIAS / PSY / TRIX。

三层校验（缺一不可）：
  1. 独立 pandas 参考实现比对（不 import 被测模块）；
  2. 覆盖 正常 / 含 NaN / 价格极小（分母极小）/ 一字板（分母为 0）/ 零价（分母为 0）五类输入；
  3. pandas_numpy / polars / sql 三后端一致性 + DuckDB 真实下推。

参考口径（与被测实现独立书写，即用户给定公式的 pandas 直译）：
  CCI  : TP=(high+low+close)/3；MA=SMA(TP,N)；MD=mean(|TP-MA|)（同一窗口）；
         CCI=(TP-MA)/(0.015*MD)；N=14。
  BIAS : (close-SMA(close,N))/SMA(close,N)*100；N=6。
  PSY  : N 窗口内 close > close_prev 的比例 * 100（严格大于；NaN 不计数）；N=12。
  TRIX : EMA1=EMA(close,N), EMA2=EMA(EMA1,N), EMA3=EMA(EMA2,N)；
         TRIX=(EMA3-prev(EMA3))/prev(EMA3)*100；N=12。

NaN / 预热策略与 RSI_WILDER / ATR_WILDER 同族：滚动窗口 ``min_periods == window``
（窗口内有效观测不足即 NaN），分母为 0 时发布 NaN（而不是 ±Inf）。
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

CANONICALS = ["cci", "bias", "psy", "trix"]
WINDOWS = {"cci": 14, "bias": 6, "psy": 12, "trix": 12}
_CCI_SCALE = 0.015


# ===========================================================================
# 独立参考实现（纯 pandas，不 import 被测模块）
# ===========================================================================
def _ref_cci(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, w: int) -> pd.DataFrame:
    tp = (high + low + close) / 3.0
    roll = tp.rolling(w, min_periods=w)
    ma = roll.mean()
    mad = roll.apply(lambda a: float(np.mean(np.abs(a - np.mean(a)))), raw=True)
    out = (tp - ma) / (_CCI_SCALE * mad.replace(0.0, np.nan))
    # 窗口内 TP 全等（一字板）→ MD = 0 → CCI 未定义 → NaN。
    # 用窗口极差判定（精确谓词），而不是 mad == 0（会被 ~1e-14 的求和残差漏掉）。
    flat = (roll.max() - roll.min()) == 0.0
    return out.where(~flat)


def _ref_bias(close: pd.DataFrame, w: int) -> pd.DataFrame:
    ma = close.rolling(w, min_periods=w).mean()
    return 100.0 * (close - ma) / ma.replace(0.0, np.nan)


def _ref_psy(close: pd.DataFrame, w: int) -> pd.DataFrame:
    """逐点显式循环：窗口内有效比较对中 close > close_prev 的占比。"""
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for j, c in enumerate(close.columns):
        x = close[c].to_numpy(dtype=float)
        n = x.size
        for i in range(w, n):
            tot = 0
            up = 0
            for k in range(i - w + 1, i + 1):
                a, b = x[k], x[k - 1]
                if np.isfinite(a) and np.isfinite(b):
                    tot += 1
                    if a > b:
                        up += 1
            if tot == w:
                out.iloc[i, j] = 100.0 * up / tot
    return out


def _ref_trix(close: pd.DataFrame, w: int) -> pd.DataFrame:
    e1 = close.ewm(span=w, adjust=False, min_periods=w).mean()
    e2 = e1.ewm(span=w, adjust=False, min_periods=w).mean()
    e3 = e2.ewm(span=w, adjust=False, min_periods=w).mean()
    prev = e3.shift(1)
    return 100.0 * (e3 - prev) / prev.replace(0.0, np.nan)


def _ref_reference(
    canonical: str, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, w: int
) -> pd.DataFrame:
    if canonical == "cci":
        return _ref_cci(high, low, close, w)
    if canonical == "bias":
        return _ref_bias(close, w)
    if canonical == "psy":
        return _ref_psy(close, w)
    if canonical == "trix":
        return _ref_trix(close, w)
    raise AssertionError(canonical)  # pragma: no cover


def _registry_calc(canonical, close, high, low, window):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"missing pandas_numpy backend for {canonical}"
    if canonical == "cci":
        return op.calculate(high, low, close, window)
    return op.calculate(close, window)


def _max_abs_err(a: pd.DataFrame, b: pd.DataFrame) -> float:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    assert av.shape == bv.shape, (av.shape, bv.shape)
    nan_a, nan_b = np.isnan(av), np.isnan(bv)
    assert np.array_equal(nan_a, nan_b), "NaN pattern mismatch"
    m = ~nan_a
    if not m.any():
        return 0.0
    return float(np.max(np.abs(av[m] - bv[m])))


# ===========================================================================
# 合成数据：normal / NaN / 极小价格 / 一字板 / 零价
# ===========================================================================
def _make_wide(
    seed: int,
    *,
    n: int = 60,
    with_nan: bool = False,
    flat_run=None,
    zero_run=None,
    tiny: bool = False,
):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A", "B"]
    base = 1e-6 if tiny else 50.0
    close = pd.DataFrame(
        {
            c: base + np.cumsum(rng.normal(0.0, base * 0.02, n)) + base * 0.25 * j
            for j, c in enumerate(cols)
        },
        index=idx,
    )
    spread = pd.DataFrame(
        {c: np.abs(rng.normal(base * 0.02, base * 0.008, n)) + base * 0.004 for c in cols},
        index=idx,
    )
    high = close + spread
    low = close - spread

    for run, price_fn in ((flat_run, "flat"), (zero_run, "zero")):
        if run is None:
            continue
        start, length = run
        for c in cols:
            price = 0.0 if price_fn == "zero" else float(close[c].iloc[start])
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
    "normal": lambda: _make_wide(101),
    "nan": lambda: _make_wide(202, with_nan=True),
    # 价格量级 1e-6：分母（MA / MAD）极小但非零，检验不产生 Inf / 精度不塌陷
    "tiny_price": lambda: _make_wide(303, tiny=True),
    # 20 连续一字板（bar 30..49）：CCI 的 MAD 恰为 0，BIAS 的 MA 非零
    "flat_board": lambda: _make_wide(404, flat_run=(30, 20)),
    # 20 连续零价（bar 30..49）：BIAS 的 MA 恰为 0，CCI 的 MAD 亦为 0
    "zero_price": lambda: _make_wide(505, zero_run=(30, 20)),
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
        assert record is not None, f"{canonical} missing from REVIEWED_MIGRATION_MANIFEST"
        assert record["review_id"] == f"R67-{canonical}"
        assert record["semantic_hash"] == f"r67-{canonical}-v1"
        assert surface.classify_canonical(canonical) == "daily"
        assert canonical in surface.daily_factor_migrated()


@pytest.mark.parametrize(
    "expr",
    [
        "cci(high, low, close, 14)",
        "bias(close, 6)",
        "psy(close, 12)",
        "trix(close, 12)",
        "cci(high, low, close, 20)",
        "bias(close, 12)",
        "psy(close, 24)",
        "trix(close, 9)",
    ],
)
def test_dsl_parse(expr):
    from factor_engine.api.dsl_parser import parse_expr

    parse_expr(expr, surface="all")
    parse_expr(expr, surface="daily")


def test_default_windows_are_documented_ones():
    for canonical, default in (("cci", 14), ("bias", 6), ("psy", 12), ("trix", 12)):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        meta = op.metadata
        assert meta.param_names[-1] == "window"
        assert meta.param_specs["window"].min <= default


# ===========================================================================
# 2) 独立参考实现比对（normal / NaN / 极小价格 / 一字板 / 零价）
# ===========================================================================
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("canonical", CANONICALS)
def test_matches_independent_reference(case, canonical):
    close, high, low = CASES[case]()
    w = WINDOWS[canonical]
    got = _registry_calc(canonical, close, high, low, w)
    ref = _ref_reference(canonical, close, high, low, w)
    err = _max_abs_err(got, ref)
    assert err < 1e-10, f"{case}/{canonical}: max abs err {err:.3e}"


def test_warmup_rows_are_nan():
    """窗口不足（含 NaN 窗口）时所有算子发布 NaN，与 RSI/ATR_WILDER 同族。"""
    close, high, low = CASES["normal"]()
    for canonical in CANONICALS:
        w = WINDOWS[canonical]
        got = _registry_calc(canonical, close, high, low, w)
        # 全 valid 数据的预热行数：cci/bias/trix 为 w-1；psy 需要 w 个比较对 → w
        warm = w if canonical == "psy" else w - 1
        assert got.iloc[:warm].isna().all().all(), f"{canonical} warmup leak"


def test_nan_windows_propagate_nan():
    """窗口内出现任一 NaN → 该窗口输出 NaN（min_periods == window）。

    ``cci`` / ``bias`` 走滚动窗口，``psy`` 走有效比较对计数，三者都是
    「窗口内有效观测不足即 NaN」。``trix`` 例外：三重 EMA 在 NaN 后按 pandas
    的 carry-forward 语义继续发布状态值（由独立参考比对覆盖）。
    """
    close, high, low = CASES["nan"]()
    for canonical in ("cci", "bias", "psy"):
        w = WINDOWS[canonical]
        got = _registry_calc(canonical, close, high, low, w)
        n = len(close.index)
        for j in range(n):
            for c in close.columns:
                if canonical == "psy":
                    # psy 的窗口是 w 个比较对，跨越 w+1 个收盘价
                    if j < w:
                        continue
                    has_nan = bool(close[c].iloc[j - w : j + 1].isna().any())
                else:
                    if j < w - 1:
                        continue
                    lo = max(0, j - w + 1)
                    # cci 用 high/low/close 三点；bias/psy 只吃 close
                    frames = (close, high, low) if canonical == "cci" else (close,)
                    has_nan = any(
                        bool(df[c].iloc[lo : j + 1].isna().any()) for df in frames
                    )
                if has_nan:
                    assert np.isnan(got.loc[close.index[j], c]), (
                        f"{canonical} {c} row {j} should be NaN"
                    )


def test_flat_board_zero_denominator():
    """一字板：CCI 的 MAD = 0 → NaN；BIAS 的 MA 非零 → 0；PSY 无上涨 → 0。"""
    close, high, low = CASES["flat_board"]()
    idx = close.index
    flat = idx[30:50]
    # 窗口整窗落入一字板的尾部区间
    cci_full = idx[30 + 14 - 1 : 50]
    bias_full = idx[30 + 6 - 1 : 50]
    psy_full = idx[30 + 12 : 50]

    assert np.array_equal(high.loc[flat].to_numpy(), low.loc[flat].to_numpy())

    cci = _registry_calc("cci", close, high, low, 14)
    assert cci.loc[cci_full].isna().all().all(), "flat board must publish NaN for cci"

    bias = _registry_calc("bias", close, high, low, 6)
    assert np.allclose(bias.loc[bias_full].to_numpy(dtype=float), 0.0, atol=1e-12)

    psy = _registry_calc("psy", close, high, low, 12)
    assert np.allclose(psy.loc[psy_full].to_numpy(dtype=float), 0.0, atol=1e-12)


def test_zero_price_zero_denominator():
    """零价：BIAS 的 MA = 0 → NaN（而非 ±Inf）；CCI 的 MAD = 0 → NaN。"""
    close, high, low = CASES["zero_price"]()
    idx = close.index
    bias_full = idx[30 + 6 - 1 : 50]
    cci_full = idx[30 + 14 - 1 : 50]

    bias = _registry_calc("bias", close, high, low, 6)
    assert bias.loc[bias_full].isna().all().all(), "zero mean must publish NaN"

    cci = _registry_calc("cci", close, high, low, 14)
    assert cci.loc[cci_full].isna().all().all()

    for canonical in CANONICALS:
        got = _registry_calc(canonical, close, high, low, WINDOWS[canonical])
        arr = got.to_numpy(dtype=float)
        assert not np.isinf(arr).any(), f"{canonical} produced ±Inf"


def test_psy_values_are_percentages():
    """PSY 值域 [0, 100]；strictly-greater 语义下恒定序列给 0，单调上升给 100。"""
    close = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]},
        index=pd.bdate_range("2024-01-02", periods=7),
    )
    up = _registry_calc("psy", close, close, close, 3)
    assert np.allclose(up.iloc[3:].to_numpy(dtype=float), 100.0)

    flat = pd.DataFrame(
        {"A": [10.0] * 7}, index=pd.bdate_range("2024-01-02", periods=7)
    )
    zero = _registry_calc("psy", flat, flat, flat, 3)
    assert np.allclose(zero.iloc[3:].to_numpy(dtype=float), 0.0)

    mixed = CASES["normal"]()[0]
    arr = _registry_calc("psy", mixed, mixed, mixed, 12).to_numpy(dtype=float)
    fin = arr[np.isfinite(arr)]
    assert fin.min() >= -1e-12 and fin.max() <= 100.0 + 1e-12


def test_bias_sign_tracks_deviation():
    """BIAS 的符号与 (close - MA) 一致。"""
    close = CASES["normal"]()[0]
    w = 6
    ma = close.rolling(w, min_periods=w).mean()
    bias = _registry_calc("bias", close, close, close, w)
    dev = (close - ma).to_numpy(dtype=float)
    val = bias.to_numpy(dtype=float)
    m = np.isfinite(dev) & np.isfinite(val)
    assert np.all(np.sign(dev[m]) == np.sign(val[m]))


def test_no_infinities_on_normal_data():
    """正常数据上四算子只输出有限值（分母为 0 时是 NaN，不是 ±Inf）。"""
    close, high, low = CASES["normal"]()
    for canonical in CANONICALS:
        arr = _registry_calc(
            canonical, close, high, low, WINDOWS[canonical]
        ).to_numpy(dtype=float)
        assert not np.isinf(arr).any(), f"{canonical} produced ±Inf"
        assert np.isfinite(arr).any(), f"{canonical} produced no finite value"


# ===========================================================================
# 3) 三后端一致性（pandas / polars / sql）
# ===========================================================================
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
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
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
    factory = make_cleaned_call_factory(canonical)
    if canonical == "cci":
        return factory(
            col(names["high"]), col(names["low"]), col(names["close"]), WINDOWS["cci"]
        )
    return factory(col(names["close"]), WINDOWS[canonical])


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
            pd_out, sql_out, check_names=False, rtol=1e-12, atol=1e-12
        )

        # 三后端同时对齐独立参考（同一 long 网格）
        w = WINDOWS[canonical]
        ref = _ref_reference(canonical, close, high, low, w)
        ref_long = ref.stack()
        ref_long.index = ref_long.index.set_names(["timestamp", "instrument"])
        ref_long = ref_long.sort_index()
        for label, out in (("pandas", pd_out), ("polars", pl_out), ("sql", sql_out)):
            pd.testing.assert_series_equal(
                out.reindex(ref_long.index),
                ref_long,
                check_names=False,
                rtol=1e-9,
                atol=1e-9,
                obj=f"{case}/{canonical}/{label}",
            )
