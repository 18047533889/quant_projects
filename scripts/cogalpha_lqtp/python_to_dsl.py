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
    lqtp_to_fe_dsl,
    translate_python,
)
from scripts.cogalpha_lqtp.dsl_sanitize import sanitize_dsl  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_dsl_compat import fe_only_operators, is_lqtp_native_dsl  # noqa: E402

# Hand-tuned DSL. Prefer LQTP operator names when equivalent (safe_div/ema/ts_quantile/ts_mean).
# Keep factor_engine-only ops (ATR/ADX/RSI/ROC/tanh/...) when LQTP has no counterpart.
EXTRA_MANUAL_DSL: dict[str, str] = {
    "factor_shadow_volume_confirmed": (
        "safe_div("
        "safe_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8)"
        " - ts_mean(safe_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8), 20),"
        " ts_std(safe_div(min(open, close) - low, min(open, close) - low + high - max(open, close) + 1e-8), 20) + 1e-8)"
        " * safe_div(volume, ts_mean(volume, 20) + 1e-8)"
    ),
    "factor_vol_asym_confirmed_range": (
        "(log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 60)))"
        " - log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 60))))"
        " * safe_div(volume, ts_mean(volume, 60))"
        " * safe_div(high - low, close)"
    ),
    "factor_vol_asym_confirmed_range_v2": (
        "(log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 60)))"
        " - log(1 + sqrt(ts_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 60))))"
        " * safe_div(volume, ts_mean(volume, 40))"
        " * safe_div(high - low, (high + low + close) / 3)"
    ),
    "factor_asym_vol_30": (
        "rank(log(safe_div("
        "ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 30),"
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 30)"
        ")))"
    ),
    "factor_asym_vol_cont_gate_30": (
        "zscore("
        "log(safe_div("
        "ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 60),"
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 60)"
        "))"
        " * cap(safe_div(volume, ts_mean(volume, 20)), 0, 3)"
        ")"
    ),
    "factor_adx_trend_vol_ema": (
        "ADX(high, low, close, 14) * sign(ts_pct(close, 10))"
        " * cap(safe_div(volume, ema(volume, 20)), 0.5, 2.0)"
    ),
    "factor_pressure_compression_simplified_v3": (
        "safe_div((close - open) * volume, ts_mean(abs((close - open) * volume), 21))"
        " * tanh(safe_div(volume, ts_mean(volume, 21)))"
        " * tanh(safe_div(ATR(high, low, close, 21), close))"
    ),
    "factor_drawdown_atr_normalized_20": (
        "-safe_div(safe_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 20))"
    ),
    "factor_drawdown_atr_normalized_v2": (
        "-safe_div(safe_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 60))"
    ),
    "factor_drawdown_atr_20_normalized": (
        "-safe_div(safe_div(close - ts_max(close, 60), ts_max(close, 60)), ATR(high, low, close, 20))"
    ),
    "factor_smoothed_atr_ratio_gated": (
        "EMA(safe_div(ATR(high, low, close, 20), ATR(high, low, close, 60)), 5)"
    ),
    "factor_resvol_volume_momentum": (
        "ROC(close, 20) * safe_div(volume, ema(volume, 20))"
    ),
    "factor_vol_lag_ret_smoothed": (
        "ema("
        "(safe_div(close, delay(close, 5)) - 1)"
        " * log(safe_div(volume, ema(volume, 20)))"
        " * (1 + safe_div(ATR(high, low, close, 20), close)),"
        " 5)"
    ),
    "factor_lagret_vol_trend_gated": (
        "ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ema(volume, 20))), 5)"
        " * where(ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5), 1, -1)"
    ),
    "factor_lagret_vol_trend_gated_v2": (
        "ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ema(volume, 20))), 5)"
        " * where(ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5), 1, -1)"
    ),
    "factor_vol_lag_ret_resvol_gated": (
        "ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ema(volume, 20))), 5)"
    ),
    "factor_asym_vol_gated_by_volume_pressure": (
        "log(cap(safe_div("
        "sqrt(ema(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 20)),"
        " sqrt(ema(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 20))),"
        " 1e-6, 1e6))"
        " * safe_div(volume, ema(volume, 20))"
    ),
    "factor_asym_intraday_sma": (
        "safe_div(ts_mean(open - low, 40), ts_mean(high - open, 40))"
    ),
    "factor_liquidity_adaptive_asym_momentum": (
        "safe_div(safe_div(close, delay(close, 21)) - 1,"
        " ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 21))"
        " / (1 + safe_div(EMA(open - low, 20), EMA(high - open, 20)))"
    ),
}

MANUAL_DSL: dict[str, str] = {
    "factor_persistence": "-ts_mean(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    "factor_persistence_ewma": "-ema(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    "factor_price_impact_stable_5d": (
        "safe_div(ts_quantile(abs(ts_pct(close, 1)), 5, 0.5), ts_mean(log(volume + 1), 5))"
    ),
    "factor_smooth_asymmetry_persistence": (
        "ema(abs(ts_pct(close, 1)), 10)"
        " * safe_div(volume, ts_mean(volume, 20))"
        " * (1 + safe_div("
        "ema(where(close > open, high - low, 0), 20),"
        " ema(where(close <= open, high - low, 0), 20) + 1e-8))"
    ),
    "factor_roughness_trend_vol_short": (
        "safe_div(ATR(high, low, close, 5), ATR(high, low, close, 20))"
        " * (safe_div(close, ts_mean(close, 20)) - 1)"
        " * safe_div(volume, ts_mean(volume, 10))"
    ),
    "factor_volatility_regime_momentum": (
        "-where(ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5), 1, -1)"
        " * ts_pct(close, 12)"
    ),
    "factor_volatility_scaled_deviation": (
        "tanh(safe_div(close - ts_mean(close, 20),"
        " ts_std(ts_pct(close, 1), 20) * ts_mean(close, 20) + 1e-8))"
    ),
    "factor_vol_regime_volume": (
        "(2 * where(ts_std(ts_pct(close, 1), 20) < ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5), 1, 0) - 1)"
        " * ts_pct(close, 10) * safe_div(volume, ts_quantile(volume, 60, 0.5))"
    ),
    "factor_corr_volume_regime": (
        "ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 20)"
        " * safe_div(volume, ema(volume, 20))"
    ),
    "factor_vol_price_coherence_v2": (
        "ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 20)"
        " * safe_div(volume, ema(volume, 20))"
    ),
    **EXTRA_MANUAL_DSL,
}

# Optional hand overrides when auto dsl_to_lqtp is insufficient.
MANUAL_LQTP_FORMULA: dict[str, str] = {}


@dataclass
class DslEntry:
    factor_id: str
    function_name: str
    dsl: str
    lqtp_formula: str
    status: str
    source: str
    notes: str = ""
    eval_route: str = "local_python"
    lqtp_native: bool = False
    fe_only_ops: str = ""


def _needs_python_route(record: dict[str, Any], *, ast_status: str = "") -> str:
    """Return a reason string when the factor should use Python materialization."""
    tools = record.get("tools", "")
    code = record.get("python_code", "")
    function_name = record.get("function_name", "")

    if ast_status == "hard":
        return "hard_python_pattern"
    if function_name in MANUAL_DSL:
        return ""
    if "style_gates=" in tools:
        return "style_gate_column"
    if "alpha_tools.classify_volume_regime" in code and (
        '== "high"' in code or "== 'high'" in code or '== "low"' in code or "== 'low'" in code
    ):
        return "volume_regime_string_compare"
    if "groupby" in code or "maximum.accumulate" in code or "cumcount" in code:
        return "hard_python_pattern"
    return ""


def _python_entry(
    *,
    factor_id: str,
    function_name: str,
    source: str,
    notes: str,
) -> DslEntry:
    return DslEntry(
        factor_id=factor_id,
        function_name=function_name,
        dsl="",
        lqtp_formula="",
        status="python",
        source=source,
        notes=notes,
        eval_route="local_python",
        lqtp_native=False,
        fe_only_ops="",
    )


def _annotate_eval_route(entry: DslEntry) -> DslEntry:
    """Route on LQTP formula when rename-compatible; keep FE-only ops from materialize DSL."""
    lqtp = (entry.lqtp_formula or entry.dsl or "").strip()
    fe = lqtp_to_fe_dsl(entry.dsl or lqtp)
    ops = fe_only_operators(fe) if fe else []
    entry.fe_only_ops = ",".join(ops)
    if entry.status == "ready" and is_lqtp_native_dsl(lqtp):
        entry.eval_route = "lqtp_dsl"
        entry.lqtp_native = True
        # Canonical catalog formula uses LQTP naming when fully compatible.
        entry.dsl = lqtp
        entry.lqtp_formula = lqtp
    else:
        # Keep factor_engine naming for local materialize; store best-effort LQTP text.
        entry.dsl = fe
        entry.lqtp_formula = lqtp
        entry.eval_route = "local_dsl" if entry.status == "ready" and entry.dsl else "local_python"
        entry.lqtp_native = False
    return entry


def _dsl_is_broken(dsl: str) -> bool:
    text = dsl.strip()
    if not text:
        return True
    if text in {"rank()", "rank(x)"}:
        return True
    if text.startswith("rank(") and text.endswith(")") and len(text) <= 8:
        return True
    return False


def _finalize_entry(
    *,
    factor_id: str,
    function_name: str,
    dsl: str,
    source: str,
    notes: str = "",
) -> DslEntry:
    dsl = sanitize_dsl(dsl)
    # Accept either FE or LQTP naming; validate against factor_engine form.
    fe_dsl = lqtp_to_fe_dsl(dsl)
    ok, msg = validate_factor_engine_dsl(fe_dsl)
    if not ok:
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source=source,
            notes=msg or notes or "dsl_invalid",
        )
    lqtp = MANUAL_LQTP_FORMULA.get(function_name) or dsl_to_lqtp(fe_dsl)
    return _annotate_eval_route(
        DslEntry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=fe_dsl,
            lqtp_formula=lqtp,
            status="ready",
            source=source,
            notes=notes,
        )
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
    py_reason = _needs_python_route(record, ast_status=result.status)
    if py_reason:
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source=result.source or "python_route",
            notes=py_reason if py_reason != "hard_python_pattern" else (result.notes or py_reason),
        )

    if not result.dsl:
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source=result.source,
            notes=result.notes or "ast_no_dsl",
        )

    entry = _finalize_entry(
        factor_id=factor_id,
        function_name=function_name,
        dsl=result.dsl,
        source=result.source,
        notes=result.notes,
    )
    if "cs_rank" in tools or "cross_sectional_transform=cs_rank" in tools:
        inner = entry.dsl.strip()
        if _dsl_is_broken(inner):
            return _python_entry(
                factor_id=factor_id,
                function_name=function_name,
                source=result.source,
                notes="cs_rank_without_inner",
            )
        entry = _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=f"rank({entry.dsl})",
            source=result.source,
            notes=result.notes,
        )
        if entry.status == "ready" and "rank(ts_mean" in entry.dsl and ">" in entry.dsl:
            return _python_entry(
                factor_id=factor_id,
                function_name=function_name,
                source=result.source,
                notes="rank_of_boolean_window",
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
    route_stats: dict[str, int] = {}
    for item in catalog:
        stats[item.status] = stats.get(item.status, 0) + 1
        route_stats[item.eval_route] = route_stats.get(item.eval_route, 0) + 1
    print(
        f"catalog: total={len(catalog)} "
        + " ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        + " routes="
        + " ".join(f"{k}={v}" for k, v in sorted(route_stats.items()))
        + f" -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
