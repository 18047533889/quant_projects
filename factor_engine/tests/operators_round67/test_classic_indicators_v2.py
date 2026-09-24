# -*- coding: utf-8 -*-
"""R67 经典技术指标 v2：Aroon / DPO / TSI / UO / Klinger / NVI / PVI / Mass /
VR / EMV / DMA / BBI / StochRSI / AD Line。

三层校验（缺一不可）：
  1. 独立参考实现比对（测试内自写，不 import 被测模块）；
  2. 覆盖 正常数据 / 含 NaN / 连续一字板（零值域）/ 零成交量 四类输入；
  3. pandas_numpy / polars / sql 三后端一致性 + DuckDB 真实下推
     （``nvi`` / ``pvi`` 为递归状态机，无 SQL 实现，只校验 pandas vs polars）。

参考口径（与被测实现独立书写）见各 ``_ref_*`` 函数。
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

SQL_BACKED: list[str] = []  # filled after SPECS

# canonical -> (DSL name, ordered panel args, positional int args)
SPECS: dict[str, tuple[str, tuple[str, ...], tuple[int, ...]]] = {
    "aroon_up": ("aroon_up", ("high", "low"), (25,)),
    "aroon_down": ("aroon_down", ("high", "low"), (25,)),
    "aroon_osc": ("aroon_osc", ("high", "low"), (25,)),
    "dpo": ("dpo", ("close",), (20,)),
    "tsi": ("tsi", ("close",), (25, 13)),
    "ultimate_osc": ("ultimate_osc", ("high", "low", "close"), (7, 14, 28)),
    "klinger": ("klinger", ("high", "low", "close", "volume"), (34, 55)),
    "nvi": ("nvi", ("close", "volume"), ()),
    "pvi": ("pvi", ("close", "volume"), ()),
    "mass_index": ("mass_index", ("high", "low"), (9, 25)),
    "vr": ("vr", ("close", "volume"), (26,)),
    "emv": ("emv", ("high", "low", "volume"), (14,)),
    "dma": ("dma", ("close",), (10, 50)),
    "bbi": ("bbi", ("close",), ()),
    "stoch_rsi": ("stoch_rsi", ("close",), (14, 14)),
    "ad_line": ("ad_line", ("high", "low", "close", "volume"), ()),
}

CANONICALS = list(SPECS)
# recursive conditional multiplicative state machines -> no finite-window SQL
NO_SQL = {"nvi", "pvi"}
SQL_BACKED = [c for c in CANONICALS if c not in NO_SQL]


# ===========================================================================
# 独立参考实现（不 import 被测模块）
# ===========================================================================
def _ref_aroon_leg(frame: pd.DataFrame, n: int, *, take_max: bool) -> pd.DataFrame:
    """显式循环：N+1 回看窗内最右极值的位置 -> 100*pos/N。"""
    W = n + 1
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype=float)
    for c in frame.columns:
        x = frame[c].to_numpy(dtype=float)
        for i in range(W - 1, x.size):
            win = x[i - W + 1 : i + 1]
            if not np.isfinite(win).all():
                continue
            best = np.max(win) if take_max else np.min(win)
            pos = max(k for k in range(W) if win[k] == best)
            out.iloc[i, out.columns.get_loc(c)] = 100.0 * pos / float(n)
    return out


def _ref_aroon(high, low, n, which):
    up = _ref_aroon_leg(high, n, take_max=True)
    if which == "aroon_up":
        return up
    dn = _ref_aroon_leg(low, n, take_max=False)
    if which == "aroon_down":
        return dn
    return up - dn


def _ref_dpo(close, n):
    ma = close.rolling(n, min_periods=n).mean()
    return close - ma.shift(n // 2 + 1)


def _ref_tsi(close, L, S):
    mom = close - close.shift(1)

    def _double(x):
        e1 = x.ewm(span=L, adjust=False, min_periods=L).mean()
        return e1.ewm(span=S, adjust=False, min_periods=S).mean()

    den = _double(mom.abs())
    return 100.0 * _double(mom) / den.replace(0.0, np.nan)


def _ref_ultimate(high, low, close, w1, w2, w3):
    pc = close.shift(1)
    true_low = np.minimum(low, pc)
    bp = close - true_low
    tr = np.maximum(high, pc) - true_low

    def _avg(w):
        num = bp.rolling(w, min_periods=w).sum()
        den = tr.rolling(w, min_periods=w).sum()
        return num / den.replace(0.0, np.nan)

    return 100.0 * (4.0 * _avg(w1) + 2.0 * _avg(w2) + _avg(w3)) / 7.0


def _ref_klinger(high, low, close, volume, w1, w2):
    hlc = high + low + close
    prev = hlc.shift(1)
    sign = pd.DataFrame(
        np.where(hlc > prev, 1.0, -1.0), index=hlc.index, columns=hlc.columns
    ).where(hlc.notna() & prev.notna())
    vf = volume * sign
    return (
        vf.ewm(span=w1, adjust=False, min_periods=w1).mean()
        - vf.ewm(span=w2, adjust=False, min_periods=w2).mean()
    )


def _ref_volume_index(close, volume, *, negative: bool) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        x = close[c].to_numpy(dtype=float)
        v = volume[c].to_numpy(dtype=float)
        row = [1000.0]
        for i in range(1, x.size):
            prev = row[-1]
            if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and np.isfinite(v[i]) and np.isfinite(v[i - 1]):
                moved = v[i] < v[i - 1] if negative else v[i] > v[i - 1]
                if moved and x[i - 1] != 0.0:
                    prev = prev * (1.0 + (x[i] - x[i - 1]) / x[i - 1])
            row.append(prev)
        out[c] = row
    return out


def _ref_mass_index(high, low, w, s):
    span = high - low
    e1 = span.ewm(span=w, adjust=False, min_periods=w).mean()
    e2 = e1.ewm(span=w, adjust=False, min_periods=w).mean()
    ratio = e1 / e2.replace(0.0, np.nan)
    return ratio.rolling(s, min_periods=s).sum()


def _ref_vr(close, volume, n):
    prev = close.shift(1)
    valid = close.notna() & prev.notna() & volume.notna()
    up = (valid & (close > prev)).to_numpy()
    dn = (valid & (close < prev)).to_numpy()
    ok = valid.to_numpy()
    vol = volume.to_numpy(dtype=float)
    up_vol = pd.DataFrame(np.where(ok, np.where(up, vol, 0.0), np.nan), index=close.index, columns=close.columns)
    dn_vol = pd.DataFrame(np.where(ok, np.where(dn, vol, 0.0), np.nan), index=close.index, columns=close.columns)
    num = up_vol.rolling(n, min_periods=n).sum()
    den = dn_vol.rolling(n, min_periods=n).sum()
    return 100.0 * num / den.replace(0.0, np.nan)


def _ref_emv(high, low, volume, n):
    mid = (high + low) / 2.0
    move = mid - mid.shift(1)
    e1 = move / (volume / 1_000_000.0).replace(0.0, np.nan)
    return e1.rolling(n, min_periods=n).mean()


def _ref_dma(close, s, l):
    return close.rolling(s, min_periods=s).mean() - close.rolling(l, min_periods=l).mean()


def _ref_bbi(close):
    m = [close.rolling(w, min_periods=w).mean() for w in (3, 6, 12, 24)]
    return (m[0] + m[1] + m[2] + m[3]) / 4.0


def _ref_stoch_rsi(close, n, rw):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / rw, adjust=False, min_periods=rw).mean()
    al = loss.ewm(alpha=1.0 / rw, adjust=False, min_periods=rw).mean()
    rs = ag / al.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((al == 0) & (ag > 0), 100.0).mask((ag == 0) & (al > 0), 0.0).mask((ag == 0) & (al == 0), 50.0)
    lo = rsi.rolling(n, min_periods=n).min()
    hi = rsi.rolling(n, min_periods=n).max()
    return 100.0 * (rsi - lo) / (hi - lo).replace(0.0, np.nan)


def _ref_ad_line(high, low, close, volume):
    span = high - low
    mfm = ((close - low) - (high - close)) / span.replace(0.0, np.nan)
    mfm = mfm.mask(span == 0.0, 0.0)
    return (mfm * volume).cumsum()


def _ref_reference(canonical, close, high, low, volume):
    fn, _panels, args = SPECS[canonical]
    if canonical.startswith("aroon"):
        return _ref_aroon(high, low, args[0], canonical)
    if canonical == "dpo":
        return _ref_dpo(close, args[0])
    if canonical == "tsi":
        return _ref_tsi(close, args[0], args[1])
    if canonical == "ultimate_osc":
        return _ref_ultimate(high, low, close, *args)
    if canonical == "klinger":
        return _ref_klinger(high, low, close, volume, *args)
    if canonical == "nvi":
        return _ref_volume_index(close, volume, negative=True)
    if canonical == "pvi":
        return _ref_volume_index(close, volume, negative=False)
    if canonical == "mass_index":
        return _ref_mass_index(high, low, args[0], args[1])
    if canonical == "vr":
        return _ref_vr(close, volume, args[0])
    if canonical == "emv":
        return _ref_emv(high, low, volume, args[0])
    if canonical == "dma":
        return _ref_dma(close, args[0], args[1])
    if canonical == "bbi":
        return _ref_bbi(close)
    if canonical == "stoch_rsi":
        return _ref_stoch_rsi(close, args[0], args[1])
    if canonical == "ad_line":
        return _ref_ad_line(high, low, close, volume)
    raise AssertionError(canonical)  # pragma: no cover


def _registry_calc(canonical, close, high, low, volume):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"missing pandas_numpy backend for {canonical}"
    named = {"close": close, "high": high, "low": low, "volume": volume}
    _fn, panels, args = SPECS[canonical]
    return op.calculate(*[named[p] for p in panels], *args)


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
# 合成数据：normal / NaN / 一字板 / 零成交量
# ===========================================================================
def _make_frames(
    seed: int,
    *,
    n: int = 70,
    with_nan: bool = False,
    flat_run=None,
    zero_vol_run=None,
):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A", "B"]
    close = pd.DataFrame(
        {c: 50.0 + np.cumsum(rng.normal(0.0, 1.0, n)) + 8.0 * j for j, c in enumerate(cols)},
        index=idx,
    )
    spread = pd.DataFrame({c: np.abs(rng.normal(1.0, 0.4, n)) + 0.2 for c in cols}, index=idx)
    high = close + spread
    low = close - spread
    volume = pd.DataFrame(
        {c: np.abs(rng.normal(1e6, 3e5, n)) + 1e5 for c in cols}, index=idx
    )
    if flat_run is not None:
        start, length = flat_run
        for c in cols:
            price = float(close[c].iloc[start])
            sl = slice(start, start + length)
            close.loc[idx[sl], c] = price
            high.loc[idx[sl], c] = price
            low.loc[idx[sl], c] = price
    if zero_vol_run is not None:
        start, length = zero_vol_run
        for c in cols:
            volume.loc[idx[start : start + length], c] = 0.0
    if with_nan:
        for c, positions in (("A", [3, 4, 12, 25, 40]), ("B", [5, 18, 33])):
            close.loc[idx[positions], c] = np.nan
        high.loc[idx[[6]], "A"] = np.nan
        low.loc[idx[[7]], "A"] = np.nan
        volume.loc[idx[[9, 44]], "A"] = np.nan
        volume.loc[idx[[11]], "B"] = np.nan
    return close, high, low, volume


CASES = {
    "normal": lambda: _make_frames(1001),
    "nan": lambda: _make_frames(2002, with_nan=True),
    # 30 连续一字板（bar 20..49）：high==low==close，Aroon 极值并列、ADL 零值域
    "flat_board": lambda: _make_frames(3003, flat_run=(20, 30)),
    # 20 连续零成交量（bar 30..49）：VR 分母为 0、EMV 除零、Klinger 力度为 0
    "zero_volume": lambda: _make_frames(4004, zero_vol_run=(30, 20)),
}


# ===========================================================================
# 1) 注册 / DSL 解析
# ===========================================================================
@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


def test_registered_with_expected_backends():
    for canonical in CANONICALS:
        entry = OperatorRegistry.catalog().get(canonical)
        assert entry is not None, f"{canonical} not in catalog"
        expected = {"pandas_numpy", "polars"}
        if canonical in SQL_BACKED:
            expected.add("sql")
        assert set(entry["backends"]) >= expected, (canonical, entry["backends"])


def test_daily_surface_migration_recorded():
    import factor_engine.cleaned_operators.operator_surface as surface

    for canonical in CANONICALS:
        record = surface.REVIEWED_MIGRATION_MANIFEST.get(canonical)
        assert record is not None, f"{canonical} missing from manifest"
        assert record["review_id"] == f"R67-{canonical}"
        assert record["semantic_hash"] == f"r67-{canonical}-v1"
        assert surface.classify_canonical(canonical) == "daily"
        assert canonical in surface.daily_factor_migrated()
        assert canonical in surface.extended_only_canonicals()


def test_aliases_resolve():
    assert OperatorRegistry.resolve_canonical("aroon") == "aroon_up"
    assert OperatorRegistry.resolve_canonical("kvo") == "klinger"
    assert OperatorRegistry.resolve_canonical("adl") == "ad_line"


@pytest.mark.parametrize(
    "expr",
    [
        "aroon_up(high, low, 25)",
        "aroon_down(high, low, 25)",
        "aroon_osc(high, low, 25)",
        "aroon(high, low, 25)",
        "dpo(close, 20)",
        "tsi(close, 25, 13)",
        "ultimate_osc(high, low, close, 7, 14, 28)",
        "klinger(high, low, close, volume, 34, 55)",
        "nvi(close, volume)",
        "pvi(close, volume)",
        "mass_index(high, low, 9, 25)",
        "vr(close, volume, 26)",
        "emv(high, low, volume, 14)",
        "dma(close, 10, 50)",
        "bbi(close)",
        "stoch_rsi(close, 14, 14)",
        "ad_line(high, low, close, volume)",
    ],
)
def test_dsl_parse(expr):
    from factor_engine.api.dsl_parser import parse_expr

    parse_expr(expr, surface="all")
    parse_expr(expr, surface="daily")


def test_no_duplicate_canonicals_registered_with_wrong_source():
    for canonical in CANONICALS:
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        assert op is not None
        assert op.metadata.param_names, canonical


# ===========================================================================
# 2) 独立参考实现比对（normal / NaN / 一字板 / 零成交量）
# ===========================================================================
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("canonical", CANONICALS)
def test_matches_independent_reference(case, canonical):
    close, high, low, volume = CASES[case]()
    got = _registry_calc(canonical, close, high, low, volume)
    ref = _ref_reference(canonical, close, high, low, volume)
    err = _max_abs_err(got, ref)
    assert err < 1e-10, f"{case}/{canonical}: max abs err {err:.3e}"


def test_warmup_rows_are_nan():
    """除状态类（nvi/pvi/ad_line：首值即定义）外，窗口不足时发布 NaN。"""
    close, high, low, volume = CASES["normal"]()
    for canonical in CANONICALS:
        if canonical in ("nvi", "pvi", "ad_line"):
            continue
        got = _registry_calc(canonical, close, high, low, volume)
        warm = _warmup_rows(canonical)
        assert warm > 0, canonical
        assert got.iloc[:warm].isna().all().all(), f"{canonical} warmup leak"


def _warmup_rows(canonical: str) -> int:
    """首个有效行下标（此前必须全 NaN）。"""
    fn, _panels, args = SPECS[canonical]
    if canonical.startswith("aroon"):
        return args[0]  # 首个 W=N+1 完整窗口落在下标 N
    if canonical == "dpo":
        return (args[0] - 1) + (args[0] // 2 + 1)
    if canonical == "tsi":
        return args[0] + args[1] - 1
    if canonical == "ultimate_osc":
        return max(args)  # prev_close 起算，需 max(window) 个有效 TR
    if canonical == "klinger":
        return args[1]
    if canonical == "mass_index":
        return 2 * (args[0] - 1) + (args[1] - 1)
    if canonical == "vr":
        return args[0]
    if canonical == "emv":
        return args[0]
    if canonical == "dma":
        return args[1] - 1
    if canonical == "bbi":
        return 23
    if canonical == "stoch_rsi":
        return args[1] + args[0] - 1
    raise AssertionError(canonical)  # pragma: no cover


def test_flat_board_aroon_endpoints():
    """一字板：极值并列取最右（当日），故 aroon_up = aroon_down = 100，osc = 0。"""
    close, high, low, volume = CASES["flat_board"]()
    up = _registry_calc("aroon_up", close, high, low, volume)
    dn = _registry_calc("aroon_down", close, high, low, volume)
    osc = _registry_calc("aroon_osc", close, high, low, volume)
    tail = close.index[49]  # 30 根一字板的末端
    assert np.allclose(up.loc[tail].to_numpy(dtype=float), 100.0)
    assert np.allclose(dn.loc[tail].to_numpy(dtype=float), 100.0)
    assert np.allclose(osc.loc[tail].to_numpy(dtype=float), 0.0)


def test_zero_volume_vr_publishes_nan():
    """下跌/上涨成交量全零的窗口 -> VR 分母为 0 -> NaN（而非 Inf）。"""
    close, high, low, volume = CASES["zero_volume"]()
    vr = _registry_calc("vr", close, high, low, volume)
    arr = vr.to_numpy(dtype=float)
    assert np.isfinite(arr).all() or np.isnan(arr).any()
    # 窗口整体落在零量区间内的行必须是 NaN
    idx = close.index
    for i, ts in enumerate(idx):
        if i < 25:
            continue
        if (volume.loc[idx[i - 25 : i]].to_numpy(dtype=float) == 0.0).all():
            assert vr.loc[ts].isna().all(), f"row {ts} should be NaN"


def test_ad_line_zero_range_contributes_zero():
    """一字板区间内 (high-low)==0 -> MFM 记 0 -> ADL 保持水平。"""
    close, high, low, volume = CASES["flat_board"]()
    adl = _registry_calc("ad_line", close, high, low, volume)
    seg = adl.loc[close.index[25:50]]
    for c in seg.columns:
        assert np.allclose(seg[c].to_numpy(dtype=float), seg[c].iloc[0]), c


def test_nan_rows_propagate():
    close, high, low, volume = CASES["nan"]()
    for canonical in ("dma", "bbi", "vr", "emv"):
        got = _registry_calc(canonical, close, high, low, volume)
        assert got.isna().any().any(), canonical


# ===========================================================================
# 3) 三后端一致性（pandas / polars / sql）
# ===========================================================================
def _to_long_source(close, high, low, volume) -> InMemorySeriesSource:
    idx = pd.MultiIndex.from_product(
        [close.index, close.columns], names=["timestamp", "instrument"]
    )
    data = {
        name: pd.Series(df.to_numpy(dtype=float).reshape(-1), index=idx)
        for name, df in (
            ("close", close),
            ("high", high),
            ("low", low),
            ("volume", volume),
        )
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


def _seed_duckdb(root: Path, close, high, low, volume) -> None:
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
                    "Volume": float(volume.loc[ts, sym]),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _duckdb_source_for(tmp_path, monkeypatch, close, high, low, volume):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
    _seed_duckdb(tmp_path / "data", close, high, low, volume)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:  # pragma: no cover
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
    fn, panels, args = SPECS[canonical]
    factory = make_cleaned_call_factory(fn)
    return factory(*[col(names[p]) for p in panels], *args)


_PANDAS_NAMES = {"close": "close", "high": "high", "low": "low", "volume": "volume"}
_DUCK_NAMES = {"close": "Close", "high": "High", "low": "Low", "volume": "Volume"}


@pytest.mark.parametrize("case", list(CASES))
def test_pandas_polars_sql_agree(case, tmp_path, monkeypatch):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    close, high, low, volume = CASES[case]()
    mem_source = _to_long_source(close, high, low, volume)
    duckdb_source = _duckdb_source_for(tmp_path, monkeypatch, close, high, low, volume)

    for canonical in CANONICALS:
        pd_out = _series(_run(mem_source, _engine_expr(canonical, _PANDAS_NAMES), "pandas"))
        pl_out = _series(_run(mem_source, _engine_expr(canonical, _PANDAS_NAMES), "polars"))
        pd.testing.assert_series_equal(
            pd_out, pl_out, check_names=False, rtol=1e-12, atol=1e-12
        )

        if canonical in SQL_BACKED:
            sql_run = _run(duckdb_source, _engine_expr(canonical, _DUCK_NAMES), "duckdb_sql")
            assert_duckdb_real_sql_execution(sql_run)
            sql_out = _series(sql_run)
            pd.testing.assert_series_equal(
                pd_out, sql_out, check_names=False, rtol=1e-11, atol=1e-11
            )

        # every backend must agree with the independent reference on the same grid
        ref = _ref_reference(canonical, close, high, low, volume)
        ref_long = ref.stack()
        ref_long.index = ref_long.index.set_names(["timestamp", "instrument"])
        ref_long = ref_long.sort_index()
        pd.testing.assert_series_equal(
            pd_out.reindex(ref_long.index), ref_long, check_names=False, rtol=1e-10, atol=1e-10
        )
