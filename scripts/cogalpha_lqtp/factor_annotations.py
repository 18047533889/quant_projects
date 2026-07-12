#!/usr/bin/env python3
"""Build per-factor formula display, Chinese rationale, and look-ahead audit."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.ast_translator import translate_python  # noqa: E402
from scripts.cogalpha_lqtp.python_to_dsl import EXTRA_MANUAL_DSL  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import LOOKAHEAD_DEFERRED_FACTORS  # noqa: E402

DEFERRED = set(LOOKAHEAD_DEFERRED_FACTORS)

THEME_PATTERNS: list[tuple[str, str]] = [
    (r"vol|atr|range|roughness|drawdown|crash|pressure", "波动/振幅类"),
    (r"volume|liquidity|vwap|dollar", "成交量/流动性类"),
    (r"momentum|trend|persistence|smooth|ewma|ema", "动量/平滑/持续性类"),
    (r"asym|shadow|shape|symmetry|imbalance|intraday", "形态/不对称/K线结构类"),
    (r"corr|coherence|regime|gate|adaptive", "相关/状态门控/自适应类"),
    (r"reversal|rank|herding|fear", "反转/排序/情绪类"),
]

RISK_LABEL = {"low": "低", "medium": "中", "high": "高", "pending_fix": "待修复"}


@dataclass
class LookaheadAudit:
    risk: str  # low | medium | high | pending_fix
    flags: list[str]
    judgment_zh: str


def _clean_formula_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^[`'\"]+|[`'\"]+$", "", t)
    return t.strip()


def _infer_theme(name: str) -> str:
    low = name.lower()
    for pat, label in THEME_PATTERNS:
        if re.search(pat, low):
            return label
    return "综合量价类"


def _is_safe_before_rank(before: str) -> bool:
    tail = before[-300:].replace("\n", " ")
    return bool(
        re.search(r"\.rolling\s*\([^)]*\)\s*$", tail)
        or re.search(r"\.expanding\s*\([^)]*\)\s*$", tail)
        or "causal_ts_rank" in tail
    )


def audit_lookahead(*, name: str, python_code: str, dsl: str) -> LookaheadAudit:
    flags: list[str] = []
    code = python_code or ""

    if name in DEFERRED:
        flags.append("deferred_full_history_rank")
    for m in re.finditer(r"\.rank\s*\(", code):
        if not _is_safe_before_rank(code[: m.start()]):
            flags.append("full_history_ts_rank")
    if re.search(r"\.rolling\s*\([^)]*center\s*=\s*True", code):
        flags.append("rolling_center_true")
    if re.search(r"\.shift\s*\(\s*-\d", code):
        flags.append("negative_shift")
    if "bfill" in code:
        flags.append("bfill_future_fill")
    if re.search(r"maximum\.accumulate|np\.cummax|\.expanding\(\)\.max", code):
        flags.append("expanding_max_causal")
    if re.search(r"groupby.*cumcount|\.cumcount\(", code):
        flags.append("drawdown_duration_groupby")
    if dsl and re.search(r"(?<!ts_)rank\s*\(", dsl) and "ts_rank" not in dsl:
        flags.append("cross_sectional_rank_dsl")

    if "deferred_full_history_rank" in flags or "full_history_ts_rank" in flags:
        risk = "pending_fix" if name in DEFERRED else "high"
    elif any(f in flags for f in ("rolling_center_true", "negative_shift", "bfill_future_fill")):
        risk = "high"
    elif flags:
        risk = "medium"
    else:
        risk = "low"

    parts: list[str] = []
    if risk == "low":
        parts.append("未发现典型未来函数模式（全样本时序 rank、center=True、负 shift 等）。")
        parts.append("因子仅使用 T 日及以前 OHLCV；RankIC 对齐 close(T)→close(T+1)，因果链完整。")
    elif risk == "pending_fix":
        parts.append("该因子在 deferred 列表中：代码里存在全历史时序 .rank()，需改为 rolling rank 后重算。")
        parts.append("当前生产 RankIC 可能偏高，暂不应作为实盘依据。")
    elif risk == "high":
        parts.append("检测到高风险算子，可能引入未来信息，建议修复后重跑。")
    else:
        parts.append("存在需注意的算子，但多数为因果实现或截面排序，需结合含义判断。")

    detail_map = {
        "full_history_ts_rank": "全历史时序 rank（会用未来样本排序）",
        "deferred_full_history_rank": "已知待修复的全历史 rank",
        "expanding_max_causal": "expanding/cummax 回撤（仅用到当前及历史，因果）",
        "drawdown_duration_groupby": "回撤持续天数 groupby（因果，但逻辑较复杂）",
        "cross_sectional_rank_dsl": "DSL 截面 rank（同日全市场排序，非时序未来函数）",
        "rolling_center_true": "rolling(center=True) 对称窗口",
        "negative_shift": "shift(-n) 引用未来",
        "bfill_future_fill": "bfill 向后填充",
    }
    if flags:
        parts.append("标记：" + "；".join(detail_map.get(f, f) for f in flags))

    return LookaheadAudit(risk=risk, flags=flags, judgment_zh=" ".join(parts))


def _formula_from_code_summary(code: str) -> str:
    """Readable pseudo-formula when AST translation fails."""
    bits: list[str] = []
    if "talib.ATR" in code:
        bits.append("ATR(high,low,close,n)")
    if "pct_change" in code or "ts_pct" in code:
        bits.append("日收益/变化率")
    if ".ewm(" in code or "talib.EMA" in code:
        bits.append("EMA/EWM 平滑")
    if ".rolling(" in code and ".rank(" in code:
        bits.append("rolling 时序分位")
    if "volume" in code and "ts_mean" not in code and "rolling" in code:
        bits.append("成交量 rolling 统计")
    if "high" in code and "low" in code:
        bits.append("(high-low) 振幅")
    if "classify_volume_regime" in code:
        bits.append("量比状态 vol/EMA(vol)")
    if "style_gate" in code:
        bits.append("风格/流动性门控")
    if "np.tanh" in code:
        bits.append("tanh 压缩")
    if not bits:
        return "见 Python 源码（复杂逻辑，无简洁 DSL）"
    return " × ".join(bits)


def build_formula_display(
    *,
    name: str,
    dsl: str,
    formula_text: str,
    python_code: str,
) -> dict[str, str]:
    dsl = (dsl or "").strip()
    if dsl:
        return {
            "primary": dsl,
            "source": "dsl_catalog",
            "note": "",
        }

    if name in EXTRA_MANUAL_DSL:
        return {
            "primary": EXTRA_MANUAL_DSL[name],
            "source": "manual_dsl",
            "note": "Python 专用因子，手写的 DSL 等价式",
        }

    tr = translate_python(python_code or "")
    if tr.status == "ready" and tr.dsl:
        return {
            "primary": tr.dsl,
            "source": "ast_translated",
            "note": "由 Python 自动翻译的 DSL 等价式（便于阅读）",
        }

    cleaned = _clean_formula_text(formula_text)
    if cleaned:
        return {
            "primary": cleaned,
            "source": "formula_text",
            "note": "来自因子文档 formula_text；含 groupby/cummax 等复杂算子时 AST 无法翻译",
        }

    return {
        "primary": _formula_from_code_summary(python_code or ""),
        "source": "code_summary",
        "note": "代码结构摘要",
    }


def _paraphrase_rationale_en(en: str) -> str:
    if not en:
        return ""
    t = en.strip()
    repl = [
        (r"\bvolatility\b", "波动率"),
        (r"\bvolume\b", "成交量"),
        (r"\bmomentum\b", "动量"),
        (r"\bmean reversion\b", "均值回归"),
        (r"\bdrawdown\b", "回撤"),
        (r"\bpersistence\b", "收益持续性"),
        (r"\basymmetry\b", "不对称性"),
        (r"\bliquidity\b", "流动性"),
        (r"\bregime\b", "市场状态"),
        (r"\brolling\b", "滚动窗口"),
        (r"\bcontrarian\b", "逆势"),
        (r"\btrend\b", "趋势"),
        (r"\bfactor\b", "因子"),
        (r"\bThis factor\b", "该因子"),
        (r"\bThe factor\b", "该因子"),
        (r"\bcaptures\b", "刻画"),
        (r"\bmeasures\b", "衡量"),
        (r"\banticipating\b", "预期"),
        (r"\bidentifying\b", "识别"),
    ]
    for pat, zh in repl:
        t = re.sub(pat, zh, t, flags=re.I)
    # Keep first ~2 sentences worth
    sents = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sents[:2]).strip()


def _design_intent_zh(name: str, code: str, formula: str, rationale_en: str) -> str:
    """Chinese design intent from code/formula patterns (not raw English paste)."""
    hints: list[str] = []
    low = (name + " " + code + " " + formula).lower()
    if "atr" in low or "high - low" in low or "range" in low:
        hints.append("用振幅/ATR 刻画波动强度")
    if "volume" in low and ("ratio" in low or "/ ema" in low or "ts_mean(volume" in low):
        hints.append("用相对成交量确认信号可信度")
    if "tanh" in low or "sigmoid" in low:
        hints.append("tanh 压缩极端值、提升稳健性")
    if "ewm" in low or "ema" in low:
        hints.append("EMA/EWM 平滑噪声")
    if "persistence" in low or "smooth" in low:
        hints.append("刻画收益路径的平滑/持续程度")
    if "asym" in low or "shadow" in low or "symmetry" in low:
        hints.append("利用 K 线上下影/不对称结构")
    if "drawdown" in low or "cummax" in low or "ts_max(close" in low:
        hints.append("结合回撤深度/持续时间")
    if "momentum" in low or "pct_change" in low or "ts_pct(close" in low:
        hints.append("引入价格动量或变化率")
    if "corr" in low or "coherence" in low:
        hints.append("量价相关/协同变化")
    if "reversal" in low or "mean reversion" in rationale_en.lower():
        hints.append("偏反转/均值回归逻辑")
    if "regime" in low or "gate" in low or "classify_volume" in low:
        hints.append("按波动/流动性状态门控")
    if "rank" in low and "rolling" in low:
        hints.append("rolling 时序分位标准化")
    if not hints:
        hints.append("基于 OHLCV 构造的日频 alpha")
    theme = _infer_theme(name)
    lead = f"属于{theme}，"
    body = "，".join(dict.fromkeys(hints)) + "。"
    if rationale_en and len(rationale_en) > 40:
        extra = _paraphrase_rationale_en(rationale_en)
        if extra and any("\u4e00" <= c <= "\u9fff" for c in extra):
            return lead + body + " 原文要点：" + extra[:280]
    return lead + body


def build_rationale_zh(
    *,
    name: str,
    theme: str,
    formula_primary: str,
    rationale_en: str,
    audit: LookaheadAudit,
    python_code: str = "",
) -> str:
    lines = [
        f"【主题】{theme}（{name}）",
        f"【公式要点】{formula_primary[:500]}{'…' if len(formula_primary) > 500 else ''}",
        f"【设计意图】{_design_intent_zh(name, python_code, formula_primary, rationale_en)}",
    ]
    risk_zh = RISK_LABEL.get(audit.risk, audit.risk)
    lines.append(f"【未来函数判断·{risk_zh}】{audit.judgment_zh}")
    return "\n".join(lines)


def build_one_annotation(
    *,
    name: str,
    catalog_entry: dict[str, Any],
    parsed_row: dict[str, Any],
) -> dict[str, Any]:
    dsl = (catalog_entry.get("dsl") or "").strip()
    py = parsed_row.get("python_code") or ""
    formula = build_formula_display(
        name=name,
        dsl=dsl,
        formula_text=parsed_row.get("formula_text") or "",
        python_code=py,
    )
    audit = audit_lookahead(name=name, python_code=py, dsl=dsl)
    theme = _infer_theme(name)
    rationale_zh = build_rationale_zh(
        name=name,
        theme=theme,
        formula_primary=formula["primary"],
        rationale_en=parsed_row.get("rationale") or "",
        audit=audit,
        python_code=py,
    )
    return {
        "factor_name": name,
        "theme": theme,
        "formula_display": formula["primary"],
        "formula_source": formula["source"],
        "formula_note": formula["note"],
        "rationale_en": parsed_row.get("rationale") or "",
        "rationale_zh": rationale_zh,
        "lookahead_risk": audit.risk,
        "lookahead_flags": audit.flags,
        "lookahead_judgment_zh": audit.judgment_zh,
        "has_dsl": bool(dsl),
        "is_python_only": not bool(dsl),
    }


def build_all_annotations(work_dir: Path) -> dict[str, Any]:
    catalog = json.loads((work_dir / "dsl_catalog.json").read_text(encoding="utf-8"))
    parsed_rows = json.loads((work_dir / "parsed_factors.json").read_text(encoding="utf-8"))
    parsed = {r["function_name"]: r for r in parsed_rows}
    out: dict[str, Any] = {}
    for entry in catalog:
        name = entry["function_name"]
        row = parsed.get(name, {})
        out[name] = build_one_annotation(name=name, catalog_entry=entry, parsed_row=row)
    meta = {
        "generated_count": len(out),
        "python_only": sum(1 for v in out.values() if v["is_python_only"]),
        "lookahead_pending_fix": sum(1 for v in out.values() if v["lookahead_risk"] == "pending_fix"),
        "lookahead_high": sum(1 for v in out.values() if v["lookahead_risk"] == "high"),
    }
    return {"meta": meta, "factors": out}


def save_annotations(work_dir: Path, data: dict[str, Any] | None = None) -> Path:
    if data is None:
        data = build_all_annotations(work_dir)
    path = work_dir / "factor_annotations.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_factor_annotation(work_dir: Path, factor_name: str) -> dict[str, Any]:
    path = work_dir / "factor_annotations.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob.get("factors", {}).get(factor_name, {})


def annotation_asdict(audit: LookaheadAudit) -> dict[str, Any]:
    return asdict(audit)
