#!/usr/bin/env python3
"""Convert CogAlpha pandas-style factor code to factor_engine DSL."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import validate_factor_engine_dsl  # noqa: E402

from scripts.cogalpha_lqtp.ast_translator import (  # noqa: E402
    TranslateResult,
    dsl_to_lqtp,
    translate_python,
)
from scripts.cogalpha_lqtp.dsl_sanitize import sanitize_dsl  # noqa: E402

# Hand-tuned DSL for patterns the AST translator still mishandles.
EXTRA_MANUAL_DSL: dict[str, str] = {
    "factor_shadow_volume_confirmed": (
        "protected_div("
        "protected_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8)"
        " - ts_mean(protected_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8), 20),"
        " ts_std(protected_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8), 20) + 1e-8)"
        " * protected_div(volume, ts_mean(volume, 20) + 1e-8)"
    ),
    "factor_vol_asym_confirmed_range": (
        "(log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 60)))"
        " - log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 60))))"
        " * protected_div(volume, ts_mean(volume, 60))"
        " * protected_div(high - low, close)"
    ),
    "factor_vol_asym_confirmed_range_v2": (
        "(log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 60)))"
        " - log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 60))))"
        " * protected_div(volume, ts_mean(volume, 40))"
        " * protected_div(high - low, (high + low + close) / 3)"
    ),
    "factor_adx_trend_vol_ema": (
        "ADX(high, low, close, 14) * sign(ts_pct(close, 10))"
        " * cap(protected_div(volume, ewm_mean(volume, 20)), 0.5, 2.0)"
    ),
    "factor_pressure_compression_simplified_v3": (
        "protected_div((close - open) * volume, ts_mean(abs((close - open) * volume), 21))"
        " * tanh(protected_div(volume, ts_mean(volume, 21)))"
        " * tanh(protected_div(ATR(high, low, close, 21), close))"
    ),
    "factor_drawdown_atr_normalized_20": (
        "-protected_div(protected_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 20))"
    ),
    "factor_drawdown_atr_normalized_v2": (
        "-protected_div(protected_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 60))"
    ),
    "factor_drawdown_atr_20_normalized": (
        "-protected_div(protected_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 20))"
    ),
    "factor_smoothed_atr_ratio_gated": (
        "EMA(protected_div(ATR(high, low, close, 20), ATR(high, low, close, 60)), 5)"
    ),
    "factor_resvol_volume_momentum": (
        "ROC(close, 20) * protected_div(volume, ewm_mean(volume, 20))"
    ),
    "factor_vol_lag_ret_smoothed": (
        "ewm_mean("
        "(protected_div(close, delay(close, 5)) - 1)"
        " * log(protected_div(volume, ewm_mean(volume, 20)))"
        " * (1 + protected_div(ATR(high, low, close, 20), close)),"
        " 5)"
    ),
    "factor_lagret_vol_trend_gated": (
        "ewm_mean((protected_div(close, delay(close, 5)) - 1) * log(protected_div(volume, ewm_mean(volume, 20))), 5)"
        " * where(ts_std(ts_pct(close, 1), 20) > ts_median(ts_std(ts_pct(close, 1), 20), 60), 1, -1)"
    ),
    "factor_lagret_vol_trend_gated_v2": (
        "ewm_mean((protected_div(close, delay(close, 5)) - 1) * log(protected_div(volume, ewm_mean(volume, 20))), 5)"
        " * where(ts_std(ts_pct(close, 1), 20) > ts_median(ts_std(ts_pct(close, 1), 20), 60), 1, -1)"
    ),
    "factor_vol_lag_ret_resvol_gated": (
        "ewm_mean((protected_div(close, delay(close, 5)) - 1) * log(protected_div(volume, ewm_mean(volume, 20))), 5)"
    ),
    "factor_asym_vol_gated_by_volume_pressure": (
        "log(cap(protected_div("
        "sqrt(ewm_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 20)),"
        " sqrt(ewm_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 20))),"
        " 1e-6, 1e6))"
        " * protected_div(volume, ewm_mean(volume, 20))"
    ),
    "factor_asym_intraday_sma": (
        "protected_div(SMA(open - low, 40), SMA(high - open, 40))"
    ),
    "factor_liquidity_adaptive_asym_momentum": (
        "protected_div(protected_div(close, delay(close, 21)) - 1,"
        " ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 21))"
        " / (1 + protected_div(EMA(open - low, 20), EMA(high - open, 20)))"
    ),
}

MANUAL_DSL: dict[str, str] = {
    "factor_persistence": "-ts_mean(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    "factor_persistence_ewma": "-ewm_mean(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    "factor_price_impact_stable_5d": (
        "protected_div(ts_median(abs(ts_pct(close, 1)), 5), ts_mean(log(add(volume, 1)), 5))"
    ),
    "factor_volatility_regime_momentum": (
        "-where(ts_std(ts_pct(close, 1), 20) > ts_median(ts_std(ts_pct(close, 1), 20), 60), 1, -1)"
        " * ts_pct(close, 12)"
    ),
    "factor_volatility_scaled_deviation": (
        "tanh(protected_div(close - ts_mean(close, 20),"
        " ts_std(ts_pct(close, 1), 20) * ts_mean(close, 20) + 1e-8))"
    ),
    "factor_vol_regime_volume": (
        "(2 * where(ts_std(ts_pct(close, 1), 20) < ts_median(ts_std(ts_pct(close, 1), 20), 60), 1, 0) - 1)"
        " * ts_pct(close, 10) * protected_div(volume, ts_median(volume, 60))"
    ),
    "factor_corr_volume_regime": (
        "ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 20)"
        " * protected_div(volume, ewm_mean(volume, 20))"
    ),
    "factor_vol_price_coherence_v2": (
        "ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 20)"
        " * protected_div(volume, ewm_mean(volume, 20))"
    ),
    **EXTRA_MANUAL_DSL,
}

MANUAL_LQTP_FORMULA: dict[str, str] = {
    "factor_vol_regime_volume": (
        "(2 * where(ts_std(close / delay(close, 1) - 1, 20) < ts_quantile(ts_std(close / delay(close, 1) - 1, 60, 0.5), 1, 0) - 1)"
        " * (close / delay(close, 10) - 1) * safe_div(volume, ts_quantile(volume, 60, 0.5))"
    ),
    "factor_corr_volume_regime": (
        "ts_corr(close / delay(close, 1) - 1, volume / delay(volume, 1) - 1, 20)"
        " * safe_div(volume, ema(volume, 20))"
    ),
    "factor_vol_price_coherence_v2": (
        "ts_corr(close / delay(close, 1) - 1, volume / delay(volume, 1) - 1, 20)"
        " * safe_div(volume, ema(volume, 20))"
    ),
}


@dataclass
class DslEntry:
    factor_id: str
    function_name: str
    dsl: str
    lqtp_formula: str
    status: str
    source: str
    notes: str = ""


def _finalize_entry(
    *,
    factor_id: str,
    function_name: str,
    dsl: str,
    source: str,
    notes: str = "",
) -> DslEntry:
    dsl = sanitize_dsl(dsl)
    ok, msg = validate_factor_engine_dsl(dsl)
    return DslEntry(
        factor_id=factor_id,
        function_name=function_name,
        dsl=dsl,
        lqtp_formula=MANUAL_LQTP_FORMULA.get(function_name, dsl_to_lqtp(dsl)),
        status="ready" if ok else "converted",
        source=source,
        notes="" if ok else msg or notes,
    )


def convert_record(record: dict[str, Any]) -> DslEntry:
    function_name = record["function_name"]
    factor_id = record["factor_id"]
    tools = record.get("tools", "")

    if function_name in MANUAL_DSL:
        return _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=MANUAL_DSL[function_name],
            source="manual",
        )

    result: TranslateResult = translate_python(record["python_code"], tools=tools)
    if not result.dsl:
        return DslEntry(
            factor_id=factor_id,
            function_name=function_name,
            dsl="",
            lqtp_formula="",
            status=result.status if result.status != "ready" else "needs_review",
            source=result.source,
            notes=result.notes,
        )

    entry = _finalize_entry(
        factor_id=factor_id,
        function_name=function_name,
        dsl=result.dsl,
        source=result.source,
        notes=result.notes,
    )
    if "cs_rank" in tools or "cross_sectional_transform=cs_rank" in tools:
        entry = _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=f"rank({entry.dsl})",
            source=result.source,
            notes=result.notes,
        )
    elif "cs_zscore" in tools or "cross_sectional_transform=cs_zscore" in tools:
        entry = _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=f"zscore({entry.dsl})",
            source=result.source,
            notes=result.notes,
        )
    return entry


def build_catalog(records: list[dict[str, Any]]) -> list[DslEntry]:
    return [convert_record(r) for r in records]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert parsed CogAlpha factors to DSL catalog")
    parser.add_argument("parsed_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.parsed_json.read_text(encoding="utf-8"))
    catalog = build_catalog(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([asdict(x) for x in catalog], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats: dict[str, int] = {}
    for item in catalog:
        stats[item.status] = stats.get(item.status, 0) + 1
    print(
        f"catalog: total={len(catalog)} "
        + " ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        + f" -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
