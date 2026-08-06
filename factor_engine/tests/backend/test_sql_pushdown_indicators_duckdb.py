# -*- coding: utf-8
"""DuckDB 端到端 parity：新增技术指标 / elementwise math SQL 下推。

通过 FactorEngine(duckdb_sql) 真实执行 DuckDB SQL，与 pandas 参考对齐。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api.columns import col
from api.cleaned_ops import make_cleaned_call_factory
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.delenv("DATA_ACCESS_CONFIG", raising=False)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    yield
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def _write_registry(tmp_path: Path, root: Path) -> Path:
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
    Open: double
    High: double
    Low: double
    Close: double
    Volume: double
    Ret: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path, n_days: int = 140) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = pd.np.random.default_rng(77) if hasattr(pd, "np") else __import__("numpy").random.default_rng(77)
    rows = []
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    for sym, base in [("A", 10.0), ("B", 20.0)]:
        close = 0.0
        for i, d in enumerate(idx):
            shock = rng.normal(0, 0.02)
            close = base + i * 0.01 + float(shock) * base
            o = close * (1.0 + rng.normal(0, 0.005))
            h = max(o, close) * (1.0 + rng.uniform(0, 0.01))
            l = min(o, close) * (1.0 - rng.uniform(0, 0.01))
            v = rng.uniform(1e4, 1e5)
            r = close / (base + (i - 1) * 0.01 + float(shock) * base) - 1.0 if i else 0.0
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Open": o,
                    "High": h,
                    "Low": l,
                    "Close": close,
                    "Volume": v,
                    "Ret": r,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _run_pair(tmp_path, monkeypatch, expr, name, *, rtol=1e-5, atol=1e-5):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")
    source = build_data_source(
        {"type": "data_access", "dataset": "test_daily", "start_date": "2023-01-02", "end_date": "2023-07-31"}
    )
    factor = Factor(name=name, expr=expr)
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
    assert diff.max() <= max(atol, rtol * aligned.iloc[:, 0].abs().max()), f"{name}: max diff {diff.max()}"


@pytest.fixture(scope="module", autouse=True)
def _load():
    from backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()


def _c(name):
    return make_cleaned_call_factory(name)


def test_duckdb_elementwise_math_match_pandas(tmp_path, monkeypatch):
    for name in ("sin", "cos", "tan", "sinh", "cosh", "log2", "log10", "square", "cube", "sigmoid", "exp_neg", "saturate", "fix"):
        _run_pair(tmp_path, monkeypatch, _c(name)(col("Ret")), name, rtol=1e-5, atol=1e-6)
    _run_pair(tmp_path, monkeypatch, _c("round")(col("Ret"), 3), "round", rtol=1e-8, atol=1e-8)


def test_duckdb_atan2_and_lerp_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("atan2")(col("High"), col("Low")), "atan2")
    _run_pair(tmp_path, monkeypatch, _c("lerp")(col("Open"), col("Close"), 0.3), "lerp")


def test_duckdb_dema_tema_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("DEMA")(col("Close"), 20), "DEMA")
    _run_pair(tmp_path, monkeypatch, _c("TEMA")(col("Close"), 15), "TEMA")


def test_duckdb_ppo_pvo_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("PPO")(col("Close"), 12, 26), "PPO")
    _run_pair(tmp_path, monkeypatch, _c("PPO_signal")(col("Close"), 12, 26, 9), "PPO_signal")
    _run_pair(tmp_path, monkeypatch, _c("PPO_hist")(col("Close"), 12, 26, 9), "PPO_hist")
    _run_pair(tmp_path, monkeypatch, _c("PVO")(col("Volume"), 12, 26), "PVO")
    _run_pair(tmp_path, monkeypatch, _c("PVO_signal")(col("Volume"), 12, 26, 9), "PVO_signal")
    _run_pair(tmp_path, monkeypatch, _c("PVO_hist")(col("Volume"), 12, 26, 9), "PVO_hist")


def test_duckdb_tsi_cmo_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("TSI")(col("Close"), 25, 13), "TSI")
    _run_pair(tmp_path, monkeypatch, _c("TSI_signal")(col("Close"), 25, 13, 9), "TSI_signal")
    _run_pair(tmp_path, monkeypatch, _c("CMO")(col("Close"), 14), "CMO")


def test_duckdb_vortex_keltner_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("VortexPlus")(col("High"), col("Low"), col("Close"), 14), "VortexPlus")
    _run_pair(tmp_path, monkeypatch, _c("VortexMinus")(col("High"), col("Low"), col("Close"), 14), "VortexMinus")
    _run_pair(tmp_path, monkeypatch, _c("KeltnerMid")(col("Close"), 20), "KeltnerMid")
    _run_pair(tmp_path, monkeypatch, _c("KeltnerUpper")(col("High"), col("Low"), col("Close"), 20, 14, 2.0), "KeltnerUpper")
    _run_pair(tmp_path, monkeypatch, _c("KeltnerLower")(col("High"), col("Low"), col("Close"), 20, 14, 2.0), "KeltnerLower")
    _run_pair(tmp_path, monkeypatch, _c("KeltnerPosition")(col("High"), col("Low"), col("Close"), 20, 14, 2.0), "KeltnerPosition")


def test_duckdb_dmi_dx_adx_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("DMI_plus")(col("High"), col("Low"), col("Close"), 14), "DMI_plus")
    _run_pair(tmp_path, monkeypatch, _c("DMI_minus")(col("High"), col("Low"), col("Close"), 14), "DMI_minus")
    _run_pair(tmp_path, monkeypatch, _c("DX")(col("High"), col("Low"), col("Close"), 14), "DX")
    _run_pair(tmp_path, monkeypatch, _c("ADX")(col("High"), col("Low"), col("Close"), 14), "ADX")


def test_duckdb_volume_flow_indicators_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("CMF")(col("High"), col("Low"), col("Close"), col("Volume"), 20), "CMF")
    _run_pair(tmp_path, monkeypatch, _c("MFI")(col("High"), col("Low"), col("Close"), col("Volume"), 14), "MFI")
    _run_pair(tmp_path, monkeypatch, _c("ADL")(col("High"), col("Low"), col("Close"), col("Volume"), 20), "ADL")
    _run_pair(tmp_path, monkeypatch, _c("ForceIndex")(col("Close"), col("Volume"), 13), "ForceIndex")
    _run_pair(tmp_path, monkeypatch, _c("EaseOfMovement")(col("High"), col("Low"), col("Volume"), 14), "EaseOfMovement")
    _run_pair(tmp_path, monkeypatch, _c("ChaikinOscillator")(col("High"), col("Low"), col("Close"), col("Volume"), 3, 10, 20), "ChaikinOscillator")


def test_duckdb_ultimate_oscillator_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("UltimateOscillator")(col("High"), col("Low"), col("Close"), 7, 14, 28), "UltimateOscillator")


def test_duckdb_candle_geometry_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("candle_body")(col("Open"), col("Close")), "candle_body")
    _run_pair(tmp_path, monkeypatch, _c("candle_abs_body")(col("Open"), col("Close")), "candle_abs_body")
    _run_pair(tmp_path, monkeypatch, _c("candle_range")(col("High"), col("Low")), "candle_range")
    _run_pair(tmp_path, monkeypatch, _c("candle_body_ratio")(col("Open"), col("High"), col("Low"), col("Close")), "candle_body_ratio")
    _run_pair(tmp_path, monkeypatch, _c("candle_upper_shadow")(col("Open"), col("High"), col("Close")), "candle_upper_shadow")
    _run_pair(tmp_path, monkeypatch, _c("candle_lower_shadow")(col("Open"), col("Low"), col("Close")), "candle_lower_shadow")
    _run_pair(tmp_path, monkeypatch, _c("candle_upper_shadow_ratio")(col("Open"), col("High"), col("Low"), col("Close")), "candle_upper_shadow_ratio")
    _run_pair(tmp_path, monkeypatch, _c("candle_lower_shadow_ratio")(col("Open"), col("High"), col("Low"), col("Close")), "candle_lower_shadow_ratio")
    _run_pair(tmp_path, monkeypatch, _c("candle_close_location")(col("High"), col("Low"), col("Close")), "candle_close_location")
    _run_pair(tmp_path, monkeypatch, _c("candle_body_position")(col("Open"), col("High"), col("Low"), col("Close")), "candle_body_position")
    _run_pair(tmp_path, monkeypatch, _c("candle_close_strength")(col("High"), col("Low"), col("Close")), "candle_close_strength")
    _run_pair(tmp_path, monkeypatch, _c("candle_rejection_upper")(col("Open"), col("High"), col("Low"), col("Close")), "candle_rejection_upper")
    _run_pair(tmp_path, monkeypatch, _c("candle_rejection_lower")(col("Open"), col("High"), col("Low"), col("Close")), "candle_rejection_lower")
    _run_pair(tmp_path, monkeypatch, _c("candle_direction")(col("Open"), col("Close")), "candle_direction")
    _run_pair(tmp_path, monkeypatch, _c("candle_gap")(col("Open"), col("Close")), "candle_gap")
    _run_pair(tmp_path, monkeypatch, _c("candle_gap_pct")(col("Open"), col("Close")), "candle_gap_pct")
    _run_pair(tmp_path, monkeypatch, _c("candle_range_atr")(col("Open"), col("High"), col("Low"), col("Close"), 14), "candle_range_atr")
    _run_pair(tmp_path, monkeypatch, _c("candle_gap_atr")(col("Open"), col("High"), col("Low"), col("Close"), 14), "candle_gap_atr")
    _run_pair(tmp_path, monkeypatch, _c("candle_overlap_ratio")(col("High"), col("Low")), "candle_overlap_ratio")
    _run_pair(tmp_path, monkeypatch, _c("candle_inside_ratio")(col("High"), col("Low")), "candle_inside_ratio")


def test_duckdb_return_decomp_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("open_close_return")(col("Open"), col("Close")), "open_close_return")
    _run_pair(tmp_path, monkeypatch, _c("limit_up_close")(col("Close"), col("High"), 0.005), "limit_up_close")
    _run_pair(tmp_path, monkeypatch, _c("limit_down_close")(col("Close"), col("Low"), 0.005), "limit_down_close")
    _run_pair(tmp_path, monkeypatch, _c("true_range")(col("High"), col("Low"), col("Close")), "true_range")


def test_duckdb_volatility_estimators_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("parkinson_vol")(col("High"), col("Low"), 20), "parkinson_vol")
    _run_pair(tmp_path, monkeypatch, _c("garman_klass_vol")(col("Open"), col("High"), col("Low"), col("Close"), 20), "garman_klass_vol")
    _run_pair(tmp_path, monkeypatch, _c("rogers_satchell_vol")(col("Open"), col("High"), col("Low"), col("Close"), 20), "rogers_satchell_vol")
    _run_pair(tmp_path, monkeypatch, _c("yang_zhang_vol")(col("Open"), col("High"), col("Low"), col("Close"), 20), "yang_zhang_vol")
    _run_pair(tmp_path, monkeypatch, _c("overnight_volatility")(col("Open"), col("Close"), 20), "overnight_volatility")
    _run_pair(tmp_path, monkeypatch, _c("intraday_volatility")(col("Open"), col("Close"), 20), "intraday_volatility")
    _run_pair(tmp_path, monkeypatch, _c("range_volatility")(col("High"), col("Low"), col("Close"), 20), "range_volatility")
    _run_pair(tmp_path, monkeypatch, _c("ulcer_index")(col("Close"), 20), "ulcer_index")
    _run_pair(tmp_path, monkeypatch, _c("high_low_spread_proxy")(col("High"), col("Low"), 20), "high_low_spread_proxy")


def test_duckdb_volume_rolling_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("ts_average_volume")(col("Volume"), 20), "ts_average_volume")
    _run_pair(tmp_path, monkeypatch, _c("volume_to_range")(col("Volume"), col("High"), col("Low"), 20), "volume_to_range")
    _run_pair(tmp_path, monkeypatch, _c("relative_volume")(col("Volume"), 20), "relative_volume")
    _run_pair(tmp_path, monkeypatch, _c("volume_zscore")(col("Volume"), 20), "volume_zscore")
    _run_pair(tmp_path, monkeypatch, _c("volume_shock")(col("Volume"), 20), "volume_shock")
    _run_pair(tmp_path, monkeypatch, _c("volume_momentum")(col("Volume"), 20), "volume_momentum")
    _run_pair(tmp_path, monkeypatch, _c("volume_volatility")(col("Volume"), 20), "volume_volatility")
    _run_pair(tmp_path, monkeypatch, _c("volume_autocorr")(col("Volume"), 20, 1), "volume_autocorr")
    _run_pair(tmp_path, monkeypatch, _c("abnormal_volume")(col("Volume"), 20), "abnormal_volume")
    _run_pair(tmp_path, monkeypatch, _c("zero_return_ratio")(col("Ret"), 20), "zero_return_ratio")


def test_duckdb_volume_acceleration_and_ratio_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("volume_acceleration")(col("Volume"), 5, 20), "volume_acceleration")
    _run_pair(tmp_path, monkeypatch, _c("up_volume_ratio")(col("Ret"), col("Volume"), 20), "up_volume_ratio")
    _run_pair(tmp_path, monkeypatch, _c("down_volume_ratio")(col("Ret"), col("Volume"), 20), "down_volume_ratio")
    _run_pair(tmp_path, monkeypatch, _c("signed_volume_imbalance")(col("Ret"), col("Volume"), 20), "signed_volume_imbalance")
    _run_pair(tmp_path, monkeypatch, _c("up_down_volume_ratio")(col("Ret"), col("Volume"), 20), "up_down_volume_ratio")
    _run_pair(tmp_path, monkeypatch, _c("volume_weighted_return")(col("Ret"), col("Volume"), 20), "volume_weighted_return")
    _run_pair(tmp_path, monkeypatch, _c("volume_weighted_momentum")(col("Close"), col("Volume"), 20), "volume_weighted_momentum")
    _run_pair(tmp_path, monkeypatch, _c("return_volume_corr")(col("Ret"), col("Volume"), 20), "return_volume_corr")
    _run_pair(tmp_path, monkeypatch, _c("return_volume_beta")(col("Ret"), col("Volume"), 20), "return_volume_beta")


def test_duckdb_vwap_obv_pvt_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("rolling_vwap")(col("Close"), col("Volume"), 20), "rolling_vwap")
    _run_pair(tmp_path, monkeypatch, _c("vwap_deviation")(col("Close"), col("Volume"), 20), "vwap_deviation")
    _run_pair(tmp_path, monkeypatch, _c("rolling_obv")(col("Close"), col("Volume"), 20), "rolling_obv")
    _run_pair(tmp_path, monkeypatch, _c("rolling_pvt")(col("Close"), col("Volume"), 20), "rolling_pvt")
    _run_pair(tmp_path, monkeypatch, _c("signed_volume")(col("Ret"), col("Volume")), "signed_volume")
    _run_pair(tmp_path, monkeypatch, _c("dollar_volume")(col("Close"), col("Volume")), "dollar_volume")
    _run_pair(tmp_path, monkeypatch, _c("adv")(col("Close"), col("Volume"), 20), "adv")
    _run_pair(tmp_path, monkeypatch, _c("amihud_illiquidity")(col("Ret"), col("Close"), col("Volume"), 20), "amihud_illiquidity")
    _run_pair(tmp_path, monkeypatch, _c("return_per_turnover")(col("Ret"), col("Volume")), "return_per_turnover")
    _run_pair(tmp_path, monkeypatch, _c("corwin_schultz_spread")(col("High"), col("Low"), 20), "corwin_schultz_spread")


def test_duckdb_band_ops_match_pandas(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("bollinger_pct_b")(col("Close"), 20, 2.0), "bollinger_pct_b")
    _run_pair(tmp_path, monkeypatch, _c("bollinger_width")(col("Close"), 20, 2.0), "bollinger_width")
    _run_pair(tmp_path, monkeypatch, _c("donchian_upper")(col("High"), 20), "donchian_upper")
    _run_pair(tmp_path, monkeypatch, _c("donchian_lower")(col("Low"), 20), "donchian_lower")
    _run_pair(tmp_path, monkeypatch, _c("donchian_mid")(col("High"), col("Low"), 20), "donchian_mid")
    _run_pair(tmp_path, monkeypatch, _c("donchian_position")(col("Close"), col("High"), col("Low"), 20), "donchian_position")
