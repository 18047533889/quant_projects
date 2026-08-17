# -*- coding: utf-8 -*-
"""Unified factor_engine DSL / Python → LQTP formula converter.

Public API
----------
- ``convert_dsl(text)`` — FE DSL → LQTP naming
- ``convert_python(code, *, name=...)`` — pandas factor body → FE DSL → LQTP
- ``convert_auto(text)`` — detect DSL vs Python
- ``convert_report_js(path)`` — batch convert ``report_data.js`` factors

CLI
---
``python -m scripts.cogalpha_lqtp.convert_to_lqtp --help``
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from scripts.cogalpha_lqtp.ast_translator import dsl_to_lqtp, lqtp_to_fe_dsl, translate_python
from scripts.cogalpha_lqtp.dsl_sanitize import sanitize_dsl
from scripts.cogalpha_lqtp.lqtp_dsl_compat import (
    fe_only_operators,
    is_lqtp_native_dsl,
    unknown_lqtp_calls,
)

Status = Literal["ready", "review", "blocked"]
SourceKind = Literal["dsl", "python", "auto"]

# Rewrites that change semantics (not just names) — surface in notes.
_APPROX_MARKERS: tuple[tuple[str, str], ...] = (
    (r"\btanh\s*\(", "tanh(x)→2*sigmoid(2*x)-1（精确恒等）"),
    (r"\bcs_rank_gaussian\s*\(", "cs_rank_gaussian→rank（丢失正态分位）"),
    (r"\bts_median\s*\(", "ts_median→ts_quantile(...,0.5)"),
    (r"\bclip\s*\(", "clip→cap"),
    (r"\band_\s*\(", "and_→infix and"),
    (r"\bor_\s*\(", "or_→infix or"),
    (r"\bnot_\s*\(", "not_→infix not"),
    (r"\bts_delay\s*\(", "ts_delay→delay"),
    (r"\b(ts_ema|ewm_mean)\s*\(", "ts_ema/ewm_mean→ema"),
    (r"\bcs_rank\s*\(", "cs_rank→rank"),
    (r"\bcs_zscore\s*\(", "cs_zscore→zscore"),
    (r"\b(protected_div|safe_div_null)\s*\(", "protected_div/safe_div_null→safe_div"),
)

# Soft-review ops that LQTP may accept but should be verified on platform.
_REVIEW_OPS = frozenset({"ts_true_streak"})


def _hl_to_span(halflife: float) -> int:
    """Map pandas ewm(halflife=h) to LQTP ema span ≈ 2/α − 1."""
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    return max(2, int(round(2.0 / alpha - 1.0)))


# Hand-verified LQTP formulas for Python factors that AST cannot cleanly rewrite.
# Keys: factor_key (report) or function_name.
MANUAL_LQTP_OVERRIDES: dict[str, tuple[str, str]] = {
    "pre_fleet:282": (
        f"ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 2.0), {_hl_to_span(7)})",
        "手写：ewm(halflife=7)→ema; median→ts_quantile(.,0.5); clip→cap",
    ),
    "factor_volume_pressure_ewm_smooth": (
        f"ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 2.0), {_hl_to_span(7)})",
        "手写：ewm(halflife=7)→ema; median→ts_quantile(.,0.5); clip→cap",
    ),
    "pre_fleet:283": (
        f"ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 2.0), {_hl_to_span(7)})",
        "手写：与 282 同核",
    ),
    "factor_volume_pressure_simplified": (
        f"ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 2.0), {_hl_to_span(7)})",
        "手写：与 ewm_smooth 同核",
    ),
    "pre_fleet:284": (
        f"0.95 * ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 3.0), {_hl_to_span(7)})"
        f" + 0.05 * ts_mean((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 3.0), 10)",
        "手写：0.95*ewm + 0.05*rolling",
    ),
    "factor_volume_pressure_light_crossover": (
        f"0.95 * ema((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 3.0), {_hl_to_span(7)})"
        f" + 0.05 * ts_mean((close / open - 1) * cap(volume / ts_quantile(volume, 10, 0.5), 0.3, 3.0), 10)",
        "手写：0.95*ewm + 0.05*rolling",
    ),
    "pre_fleet:281": (
        f"ema((close / open - 1) * (volume / ts_quantile(volume, 20, 0.5)), {_hl_to_span(5)})",
        "手写：ewm(halflife=5)→ema",
    ),
    "factor_volume_pressure_imbalance": (
        f"ema((close / open - 1) * (volume / ts_quantile(volume, 20, 0.5)), {_hl_to_span(5)})",
        "手写：ewm(halflife=5)→ema",
    ),
    "pre_fleet:293": (
        f"ema(ts_pct(close, 1) * (volume / ts_quantile(volume, 30, 0.5)), {_hl_to_span(5)})",
        "手写：日收益 ts_pct(close,1)",
    ),
    "factor_volume_confirmed_return": (
        f"ema(ts_pct(close, 1) * (volume / ts_quantile(volume, 30, 0.5)), {_hl_to_span(5)})",
        "手写：日收益 ts_pct(close,1)",
    ),
    "w01:239": (
        "rank(ts_true_streak(volume > 2 * ts_quantile(volume, 50, 0.5)))",
        "手写近似：原 Python 对超阈强度累加；此处用连续高量长度",
    ),
    "factor_volume_intensity_streak": (
        "rank(ts_true_streak(volume > 2 * ts_quantile(volume, 50, 0.5)))",
        "手写近似：原 Python 对超阈强度累加；此处用连续高量长度",
    ),
    # --- weekly python-9 (requested_factors_dsl_dump) ---
    "cand_vw_ew_spread_responsive": (
        "safe_div(ema(ts_pct(close,1)*volume,19),ema(volume,19))-ema(ts_pct(close,1),19)",
        "手写：ewm(alpha=0.1)→ema(span=19)；VW−EW return spread",
    ),
    "factor_vw_ew_spread_responsive": (
        "safe_div(ema(ts_pct(close,1)*volume,19),ema(volume,19))-ema(ts_pct(close,1),19)",
        "手写：ewm(alpha=0.1)→ema(span=19)；VW−EW return spread",
    ),
    "cand_overnight_intraday_divergence_simplified": (
        "ema(safe_div(close,open)-1,5)-ema(safe_div(open,delay(close,1))-1,5)",
        "手写：intraday−overnight ema(5) divergence",
    ),
    "factor_overnight_intraday_divergence_simplified": (
        "ema(safe_div(close,open)-1,5)-ema(safe_div(open,delay(close,1))-1,5)",
        "手写：intraday−overnight ema(5) divergence",
    ),
    "cand_fusion_upcap_volregime": (
        "safe_div(ts_sum(volume*where(ts_pct(close,1)>0,1,0),20),ts_sum(volume,20))"
        "*(1+(safe_div(volume,ema(volume,20))-1)*0.5)",
        "手写：up-volume capture × vol_regime",
    ),
    "factor_fusion_upcap_volregime": (
        "safe_div(ts_sum(volume*where(ts_pct(close,1)>0,1,0),20),ts_sum(volume,20))"
        "*(1+(safe_div(volume,ema(volume,20))-1)*0.5)",
        "手写：up-volume capture × vol_regime",
    ),
    "cand_gap_reversal_intensity": (
        "ts_sum((-where(safe_div(open,delay(close,1))-1<0,safe_div(open,delay(close,1))-1,0))"
        "*where(safe_div(close,open)-1>0,safe_div(close,open)-1,0),5)",
        "手写：min/max→where；5d gap-reversal intensity",
    ),
    "factor_gap_reversal_intensity": (
        "ts_sum((-where(safe_div(open,delay(close,1))-1<0,safe_div(open,delay(close,1))-1,0))"
        "*where(safe_div(close,open)-1>0,safe_div(close,open)-1,0),5)",
        "手写：min/max→where；5d gap-reversal intensity",
    ),
    "cand_gpdev_volregime_ema10": (
        "ema((safe_div(close,sqrt(high*low))-1)*(safe_div(volume,ema(volume,20))-1),10)",
        "手写：geo-mean deviation × vol_strength, ema10",
    ),
    "factor_gpdev_volregime_ema10": (
        "ema((safe_div(close,sqrt(high*low))-1)*(safe_div(volume,ema(volume,20))-1),10)",
        "手写：geo-mean deviation × vol_strength, ema10",
    ),
    "cand_gap_vol_cluster_adaptive_w30": (
        "(2*sigmoid(2*(((safe_div(close,open)-1)-(safe_div(open,delay(close,1))-1))"
        "*safe_div(volume,ema(volume,20))*10))-1)"
        "*(1+0.5*(2*sigmoid(2*((-ts_corr(power(ts_pct(close,1),2),"
        "delay(power(ts_pct(close,1),2),1),30))*2))-1))",
        "手写：tanh→sigmoid 恒等；sq_ret→power；cluster weight",
    ),
    "factor_gap_vol_cluster_adaptive_w30": (
        "(2*sigmoid(2*(((safe_div(close,open)-1)-(safe_div(open,delay(close,1))-1))"
        "*safe_div(volume,ema(volume,20))*10))-1)"
        "*(1+0.5*(2*sigmoid(2*((-ts_corr(power(ts_pct(close,1),2),"
        "delay(power(ts_pct(close,1),2),1),30))*2))-1))",
        "手写：tanh→sigmoid 恒等；sq_ret→power；cluster weight",
    ),
    "cand_reversal_volregime_overnight_csz": (
        "(-ts_pct(close,3)*safe_div("
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),5),"
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),20)))"
        "*(1/(1+safe_div(abs(safe_div(open,delay(close,1))-1),"
        "ts_std(safe_div(open,delay(close,1))-1,20))))",
        "手写：TR where-max；atr_ratio×−ret3d×overnight attenuation",
    ),
    "factor_reversal_volregime_overnight_csz": (
        "(-ts_pct(close,3)*safe_div("
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),5),"
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),20)))"
        "*(1/(1+safe_div(abs(safe_div(open,delay(close,1))-1),"
        "ts_std(safe_div(open,delay(close,1))-1,20))))",
        "手写：TR where-max；atr_ratio×−ret3d×overnight attenuation",
    ),
    "cand_vol_expansion_atr_ratio": (
        "sign(close-open)*sigmoid(5*(safe_div("
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),5),"
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),60))-1))",
        "手写近似：talib ATR→ts_mean(TR)；sigmoid expansion × direction",
    ),
    "factor_vol_expansion_atr_ratio": (
        "sign(close-open)*sigmoid(5*(safe_div("
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),5),"
        "ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),"
        "abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,"
        "where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),"
        "abs(low-delay(close,1)))),60))-1))",
        "手写近似：talib ATR→ts_mean(TR)；sigmoid expansion × direction",
    ),
    "cand_geometric_deviation_vol_asym_tilt": (
        "ema(sign(((2*log(cap(close,1.0e-12,1.0e18))-log(cap(high,1.0e-12,1.0e18))"
        "-log(cap(low,1.0e-12,1.0e18)))/3))"
        "*power(((2*log(cap(close,1.0e-12,1.0e18))-log(cap(high,1.0e-12,1.0e18))"
        "-log(cap(low,1.0e-12,1.0e18)))/3),2),5)"
        "*(1+where(safe_div("
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),"
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)"
        ")>0,safe_div("
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),"
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)"
        "),0))",
        "手写：geo log-dev^2 ema5 × upside vol-asym tilt",
    ),
    "factor_geometric_deviation_vol_asym_tilt": (
        "ema(sign(((2*log(cap(close,1.0e-12,1.0e18))-log(cap(high,1.0e-12,1.0e18))"
        "-log(cap(low,1.0e-12,1.0e18)))/3))"
        "*power(((2*log(cap(close,1.0e-12,1.0e18))-log(cap(high,1.0e-12,1.0e18))"
        "-log(cap(low,1.0e-12,1.0e18)))/3),2),5)"
        "*(1+where(safe_div("
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),"
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)"
        ")>0,safe_div("
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),"
        "ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)"
        "+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)"
        "),0))",
        "手写：geo log-dev^2 ema5 × upside vol-asym tilt",
    ),
}


def _lookup_manual(*keys: str) -> tuple[str, str] | None:
    for key in keys:
        if key and key in MANUAL_LQTP_OVERRIDES:
            return MANUAL_LQTP_OVERRIDES[key]
    return None


@dataclass
class ConvertResult:
    """One formula conversion outcome."""

    input: str
    source_kind: SourceKind
    fe_formula: str = ""
    lqtp_formula: str = ""
    status: Status = "blocked"
    lqtp_native: bool = False
    notes: list[str] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)
    fe_only_ops: list[str] = field(default_factory=list)
    unknown_ops: list[str] = field(default_factory=list)
    factor_key: str = ""
    factor_name: str = ""
    fitness_direction: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_fitness_direction(formula: str, direction: int | None) -> tuple[str, bool]:
    """Bake fitness_direction into the formula so submitters don't need a separate sign.

    When ``direction == -1``, wraps as ``-(...)``.
    """
    text = (formula or "").strip()
    if not text or direction is None or int(direction) >= 0:
        return text, False
    return f"-({text})", True


def sanitize_lqtp_clickhouse_literals(formula: str) -> str:
    """Avoid ClickHouse ``NO_COMMON_TYPE`` (Float64 vs UInt64) inside ``cap``/``greatest``.

    Bare integers like ``1000000000000`` become UInt64; mixed with float bounds
    (``1.0e-12``) fails. Rewrite large integer literals to scientific floats.
    Also normalize long decimal epsilons to ``1.0e-12`` style.
    """
    text = formula or ""
    if not text:
        return text

    def _sci(n: float) -> str:
        s = f"{n:.12g}"
        # Prefer compact scientific for extremes.
        if abs(n) >= 1e9 or (n != 0 and abs(n) < 1e-6):
            s = f"{n:.0e}" if abs(n) >= 1 else f"{n:.0e}"
            s = s.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
            # 1e+12 → 1e12
            s = re.sub(r"e\+0*", "e", s)
            s = re.sub(r"e-0*", "e-", s)
        return s

    # long decimal → scientific (e.g. 0.000000000001)
    def _dec_repl(m: re.Match[str]) -> str:
        return _sci(float(m.group(0)))

    text = re.sub(r"(?<![\w.])0\.\d{6,}(?![\w.])", _dec_repl, text)

    # large integer literals (≥10 digits) → scientific float
    def _int_repl(m: re.Match[str]) -> str:
        return _sci(float(int(m.group(0))))

    text = re.sub(r"(?<![\w.])\d{10,}(?![\w.])", _int_repl, text)
    return text


def _detect_kind(text: str) -> SourceKind:
    raw = (text or "").strip()
    if not raw:
        return "dsl"
    if re.search(r"^\s*def\s+\w+\s*\(", raw, re.M) or "df[" in raw or "df." in raw:
        return "python"
    if "import " in raw and ("pandas" in raw or "numpy" in raw or "np." in raw):
        return "python"
    return "dsl"


def _collect_approximations(src: str) -> list[str]:
    notes: list[str] = []
    for pattern, msg in _APPROX_MARKERS:
        if re.search(pattern, src or ""):
            notes.append(msg)
    return notes


def _finalize(
    *,
    original: str,
    source_kind: SourceKind,
    fe_formula: str,
    lqtp_formula: str,
    notes: list[str] | None = None,
    factor_key: str = "",
    factor_name: str = "",
) -> ConvertResult:
    notes = list(notes or [])
    approximations = _collect_approximations(original) + _collect_approximations(fe_formula)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in approximations:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    approximations = deduped

    # Detect FE-only / unknown on the *LQTP* text (post rewrite).
    fe_only = fe_only_operators(lqtp_formula)
    unknown = unknown_lqtp_calls(lqtp_formula)
    native = bool(lqtp_formula) and is_lqtp_native_dsl(lqtp_formula)

    status: Status
    if not lqtp_formula:
        status = "blocked"
        if not notes:
            notes.append("无法生成 LQTP 公式")
    elif fe_only:
        status = "blocked"
        notes.append("FE-only 算子: " + ", ".join(fe_only))
    elif unknown:
        status = "review"
        notes.append("未知 LQTP 调用: " + ", ".join(unknown))
    elif any("丢失" in a for a in approximations):
        status = "review"
        notes.append("存在语义近似，建议平台验证")
    elif any(op in lqtp_formula for op in _REVIEW_OPS):
        status = "ready"
        notes.append("含 ts_true_streak，请确认 LQTP 已启用该算子")
    elif native:
        status = "ready"
    else:
        status = "review"
        notes.append("未能确认为 LQTP native，请人工检查")

    return ConvertResult(
        input=original,
        source_kind=source_kind,
        fe_formula=fe_formula,
        lqtp_formula=lqtp_formula,
        status=status,
        lqtp_native=native and status in {"ready", "review"},
        notes=notes,
        approximations=approximations,
        fe_only_ops=fe_only,
        unknown_ops=unknown,
        factor_key=factor_key,
        factor_name=factor_name,
    )


def convert_dsl(
    text: str,
    *,
    factor_key: str = "",
    factor_name: str = "",
    validate_fe: bool = True,
) -> ConvertResult:
    """Convert factor_engine DSL text to LQTP formula."""
    original = (text or "").strip()
    if not original:
        return _finalize(
            original="",
            source_kind="dsl",
            fe_formula="",
            lqtp_formula="",
            notes=["空公式"],
            factor_key=factor_key,
            factor_name=factor_name,
        )

    manual = _lookup_manual(factor_key, factor_name)
    if manual is not None:
        lqtp, note = manual
        return _finalize(
            original=original,
            source_kind="dsl",
            fe_formula=sanitize_dsl(lqtp_to_fe_dsl(lqtp)),
            lqtp_formula=sanitize_lqtp_clickhouse_literals(lqtp),
            notes=[note, "来源: MANUAL_LQTP_OVERRIDES"],
            factor_key=factor_key,
            factor_name=factor_name,
        )

    fe = sanitize_dsl(lqtp_to_fe_dsl(original))
    notes: list[str] = []
    if validate_fe:
        try:
            from api.mining_integration import validate_factor_engine_dsl

            ok, msg = validate_factor_engine_dsl(fe)
            if not ok:
                notes.append(f"FE 校验未通过: {msg}")
        except Exception as exc:  # pragma: no cover - optional when FE not on path
            notes.append(f"跳过 FE 校验: {exc}")

    lqtp = sanitize_lqtp_clickhouse_literals(dsl_to_lqtp(fe))
    result = _finalize(
        original=original,
        source_kind="dsl",
        fe_formula=fe,
        lqtp_formula=lqtp,
        notes=notes,
        factor_key=factor_key,
        factor_name=factor_name,
    )
    # If FE validation failed hard, demote unless still native LQTP text.
    if any(n.startswith("FE 校验未通过") for n in result.notes) and result.status == "ready":
        result.status = "review"
    return result


def convert_python(
    code: str,
    *,
    name: str = "",
    tools: str = "",
    factor_key: str = "",
    factor_name: str = "",
    validate_fe: bool = True,
) -> ConvertResult:
    """Convert pandas-style Python factor body to LQTP (via FE DSL)."""
    original = (code or "").strip()
    if not original:
        return _finalize(
            original="",
            source_kind="python",
            fe_formula="",
            lqtp_formula="",
            notes=["空 Python 代码"],
            factor_key=factor_key,
            factor_name=factor_name or name,
        )

    manual = _lookup_manual(factor_key, name, factor_name)
    if manual is not None:
        lqtp, note = manual
        return _finalize(
            original=original,
            source_kind="python",
            fe_formula="",
            lqtp_formula=lqtp,
            notes=[note, "来源: MANUAL_LQTP_OVERRIDES"],
            factor_key=factor_key,
            factor_name=factor_name or name,
        )

    translated = translate_python(original, tools=tools)
    if translated.status in {"hard", "needs_review"} or not translated.dsl:
        return _finalize(
            original=original,
            source_kind="python",
            fe_formula="",
            lqtp_formula="",
            notes=[
                translated.notes or translated.status or "python 无法自动转 DSL",
                "请手写 LQTP 公式（写入 MANUAL_LQTP_OVERRIDES）或走本地落值上传",
            ],
            factor_key=factor_key,
            factor_name=factor_name or name,
        )

    # Reuse DSL path for naming + validation.
    mid = convert_dsl(
        translated.dsl,
        factor_key=factor_key,
        factor_name=factor_name or name,
        validate_fe=validate_fe,
    )
    mid.input = original
    mid.source_kind = "python"
    if translated.notes:
        mid.notes = [translated.notes, *mid.notes]
    mid.notes = [f"python→dsl via {translated.source}", *mid.notes]
    return mid


def convert_auto(
    text: str,
    *,
    factor_key: str = "",
    factor_name: str = "",
    validate_fe: bool = True,
) -> ConvertResult:
    """Auto-detect DSL vs Python and convert."""
    kind = _detect_kind(text)
    if kind == "python":
        return convert_python(
            text,
            factor_key=factor_key,
            factor_name=factor_name,
            validate_fe=validate_fe,
        )
    return convert_dsl(
        text,
        factor_key=factor_key,
        factor_name=factor_name,
        validate_fe=validate_fe,
    )


def _load_report_js(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if "REPORT_DATA" in text and "=" in text[:80]:
        payload = text[text.find("=") + 1 :].strip()
        if payload.endswith(";"):
            payload = payload[:-1]
        return json.loads(payload)
    return json.loads(text)


def convert_report_factors(
    report_js: Path | str,
    *,
    validate_fe: bool = False,
) -> dict[str, Any]:
    """Batch-convert factors from a ``report_data.js`` payload."""
    path = Path(report_js)
    data = _load_report_js(path)
    factors = data.get("factors") or []
    results: list[ConvertResult] = []
    for item in factors:
        key = str(item.get("factor_key") or item.get("factor_id") or "")
        name = str(item.get("factor_name") or key)
        payload = str(item.get("source_payload") or "").strip()
        field = str(item.get("payload_field") or "")
        route = str(item.get("route") or "")
        raw_dir = item.get("fitness_direction")
        try:
            direction: int | None = int(raw_dir) if raw_dir is not None else None
        except (TypeError, ValueError):
            direction = None

        if field == "python_code" or route == "python":
            result = convert_python(
                payload,
                factor_key=key,
                factor_name=name,
                validate_fe=validate_fe,
            )
        else:
            result = convert_dsl(
                payload,
                factor_key=key,
                factor_name=name,
                validate_fe=validate_fe,
            )

        result.fitness_direction = direction
        if result.lqtp_formula:
            result.lqtp_formula = sanitize_lqtp_clickhouse_literals(result.lqtp_formula)
        if result.lqtp_formula and direction == -1:
            flipped, applied = apply_fitness_direction(result.lqtp_formula, direction)
            if applied:
                result.lqtp_formula = flipped
                result.notes = [
                    "fitness_direction=-1 → 公式已加前缀负号 -(...)",
                    *result.notes,
                ]
        results.append(result)

    summary = {
        "total": len(results),
        "ready": sum(1 for r in results if r.status == "ready"),
        "review": sum(1 for r in results if r.status == "review"),
        "blocked": sum(1 for r in results if r.status == "blocked"),
    }
    return {
        "schema_version": "1.0",
        "source": str(path),
        "manual_ref": "LQTP 因子服务用户手册 2026-07-19",
        "summary": summary,
        "naming_rules": [
            "clip → cap",
            "ts_delay → delay",
            "ts_ema / ewm_mean → ema",
            "cs_rank → rank",
            "cs_zscore → zscore",
            "and_ / or_ / not_ → infix and / or / not",
            "tanh(x) → 2*sigmoid(2*x)-1",
            "cs_rank_gaussian(x) → rank(x) (approx)",
            "ts_median(x,d) → ts_quantile(x,d,0.5)",
            "safe_div / coalesce / where / ts_* 保持同名",
            "fitness_direction=-1 → 公式前加 -(...)",
        ],
        "factors": [r.to_dict() for r in results],
    }


def render_markdown(batch: dict[str, Any]) -> str:
    """Render batch conversion result as a copy-paste submission markdown."""
    summary = batch.get("summary") or {}
    lines = [
        "# LQTP 提交公式（可直接复制）",
        "",
        f"> 来源 `{batch.get('source')}` · ready={summary.get('ready')} / "
        f"review={summary.get('review')} / blocked={summary.get('blocked')}",
        "",
        "每个因子两行可复制字段：",
        "",
        "- **因子名**：`name` → LQTP 提交时的因子命名",
        "- **公式**：`formula` → 已按 LQTP 平台算子命名（`cap`/`delay`/`ema`/`rank`/`zscore`/…）",
        "",
        "## 命名对齐",
        "",
    ]
    for rule in batch.get("naming_rules") or []:
        lines.append(f"- `{rule}`")

    lines += [
        "",
        "## 一览",
        "",
        "| # | 因子名 | key | dir | status |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(batch.get("factors") or [], 1):
        name = (row.get("factor_name") or "").replace("|", "\\|")
        key = (row.get("factor_key") or "").replace("|", "\\|")
        direction = row.get("fitness_direction")
        dir_s = "" if direction is None else str(direction)
        lines.append(f"| {i} | `{name}` | `{key}` | {dir_s} | {row.get('status')} |")

    lines += ["", "## 提交清单（全量公式）", ""]
    for i, row in enumerate(batch.get("factors") or [], 1):
        name = row.get("factor_name") or row.get("factor_key") or f"factor_{i}"
        key = row.get("factor_key") or ""
        status = row.get("status") or ""
        formula = row.get("lqtp_formula") or ""
        direction = row.get("fitness_direction")
        notes = "; ".join(row.get("notes") or [])
        lines += [
            f"### {i}. `{name}`",
            "",
            f"- key: `{key}`",
            f"- fitness_direction: `{direction}`",
            f"- status: **{status}**",
        ]
        if notes:
            lines.append(f"- note: {notes}")
        lines += [
            "",
            "**因子名（复制）**",
            "",
            "```",
            name,
            "```",
            "",
            "**LQTP 公式（复制）**",
            "",
            "```",
            formula,
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"
