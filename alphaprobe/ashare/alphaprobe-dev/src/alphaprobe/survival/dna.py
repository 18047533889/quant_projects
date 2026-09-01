"""Factor DNA 提取器（任务书 §46 / Phase 10）。

给每个 factor 建立 DNA（field families / operators / mechanisms / horizon /
complexity），供 §47 的生存统计做「operator 与字段、horizon、complexity 强
混杂」下的分组分析。纯统计/规则代码，不解析 AST（公式为文本）。

factor_engine 侧提供可选的 ``validate_factor_engine_dsl`` 验证；无该包时
降级为宽松文本抽取，保证测试环境无依赖可跑。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from alphaprobe.contracts import FactorDNA

__all__ = [
    "FieldFamilyMap",
    "MECHANISM_RULES",
    "field_families_of",
    "extract_operator_multiset",
    "max_window_of",
    "complexity_bucket_of",
    "dna_from_formula",
]

# ---------------------------------------------------------------------------
# §46.1 Data DNA：field → family 映射（允许多标签）
# ---------------------------------------------------------------------------

#: 精确字段名 → family（优先级最高）
FIELD_FAMILY_BY_NAME: dict[str, str] = {
    "open": "price_volume",
    "high": "price_volume",
    "low": "price_volume",
    "close": "price_volume",
    "pre_close": "price_volume",
    "vwap": "price_volume",
    "volume": "price_volume",
    "amount": "price_volume",
    "turn": "price_volume",
    "turnover": "price_volume",
    "change": "price_volume",
    "pct_change": "price_volume",
    "ret": "price_volume",
    "return": "price_volume",
    "returns": "price_volume",
    "pe": "valuation",
    "pe_ttm": "valuation",
    "pb": "valuation",
    "pb_lf": "valuation",
    "pcf": "valuation",
    "pcf_ocf": "valuation",
    "ps": "valuation",
    "ps_ttm": "valuation",
    "market_cap": "valuation",
    "total_mv": "valuation",
    "circ_mv": "valuation",
    "mv": "valuation",
}

#: §46.1 多标签：字段除主 family 外的次级 family（volume/amount 兼属 liquidity）
FIELD_SECONDARY_FAMILY: dict[str, str] = {
    "volume": "liquidity",
    "amount": "liquidity",
    "turn": "liquidity",
    "turnover": "liquidity",
}

#: 子串 → family（仅当未被精确名覆盖时兜底）
_FIELD_FAMILY_BY_SUBSTRING: list[tuple[str, str]] = [
    ("market_cap", "valuation"),
    ("total_mv", "valuation"),
    ("circ_mv", "valuation"),
    ("_mv", "valuation"),
    ("pe", "valuation"),
    ("pb", "valuation"),
    ("pcf", "valuation"),
    ("ps", "valuation"),
    ("roe", "fundamental"),
    ("roa", "fundamental"),
    ("eps", "fundamental"),
    ("gross_margin", "fundamental"),
    ("net_margin", "fundamental"),
    ("profit", "fundamental"),
    ("earnings", "fundamental"),
    ("revenue", "fundamental"),
    ("income", "fundamental"),
    ("equity", "fundamental"),
    ("debt", "fundamental"),
    ("asset", "fundamental"),
    ("liability", "fundamental"),
    ("cf", "fundamental"),
    ("cashflow", "fundamental"),
    ("cash_flow", "fundamental"),
    ("operating", "fundamental"),
    ("intraday", "intraday"),
    ("industry", "industry"),
    ("sector", "industry"),
    ("index", "index"),
]

#: 正则字段 token：字母/数字/下划线，支持常用大小写变体（Price/Close 等）
_FIELD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: family 集合：同值视为同一 family（hash）
FieldFamilyMap = dict[str, str]


def field_families_of(formula: str) -> list[str]:
    """§46.1：从公式文本抽取 field family（多标签，按首次出现顺序去重）。

    - close/volume/amount/vwap 等 → price_volume；
    - pe/pb/market_cap 等 → valuation；
    - 其余匹配 fundamental / intraday / industry / index 子串。
    """
    text = str(formula)
    names = [m.group(0) for m in _FIELD_TOKEN_RE.finditer(text)]
    lower_hits: list[str] = []
    exact_hits: list[str] = []
    secondary_hits: list[str] = []
    for name in names:
        lower = name.lower()
        fam = FIELD_FAMILY_BY_NAME.get(lower)
        if fam is not None:
            exact_hits.append(fam)
            sec = FIELD_SECONDARY_FAMILY.get(lower)
            if sec is not None:
                secondary_hits.append(sec)
            continue
        for sub, f in _FIELD_FAMILY_BY_SUBSTRING:
            if sub in lower:
                lower_hits.append(f)
                break
    # 去重保序（次级标签排最后，主 family 优先）
    out: list[str] = []
    for f in exact_hits + lower_hits + secondary_hits:
        if f not in out:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# §46.2 Operator DNA：operator multiset
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _extract_operator_names(text: str) -> tuple[str, ...]:
    return tuple(m.group(1) for m in _CALL_RE.finditer(text))


def extract_operator_multiset(
    formula: str,
    *,
    allowlist: "set[str] | frozenset[str] | None" = None,
    validate: bool = False,
) -> Counter[str]:
    """§46.2：operator multiset（函数名多重集，按出现次数计）。

    未提供 allowlist 时取全部 ``name(`` 形式的 token；给定 allowlist 时只保留
    在 allowlist 中的算子（factor_engine DSL 算子面）。``validate`` 开启时会
    先尝试 factor_engine DSL 验证，失败返回空 multiset。
    """
    text = str(formula)
    if validate:
        try:
            from alphaprobe.fe_bridge.dsl_expression import validate_factor_engine_dsl

            ok, _ = validate_factor_engine_dsl(text)
        except Exception:  # factor_engine 不可用 → 跳过验证
            ok = True
        if not ok:
            return Counter()
    names = _extract_operator_names(text)
    if allowlist is not None:
        names = [n for n in names if n in allowlist]
    return Counter(names)


def default_operator_allowlist() -> set[str] | None:
    """尝试从 factor_engine 注册表构建 DSL 算子白名单；不可用时返回 None。

    None 表示「未限制」（宽松模式），由调用方决定是否过滤。
    """
    try:
        from alphaprobe.fe_bridge.dsl_expression import extract_dsl_operator_names
        from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

        ensure_factor_engine_importable()
        from factor_engine.api.operator_registry import build_dsl_allowlist

        allow = build_dsl_allowlist()
        if isinstance(allow, dict):
            return set(allow)
        return set(allow)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# §46.4 Temporal DNA：horizon bucket
# ---------------------------------------------------------------------------

_WINDOW_RE = re.compile(r"\b(\d{1,4})\b")


def max_window_of(formula: str) -> int:
    """公式中出现的最大窗口参数（ts_* 等算子的第二个数字参数）。

    对 ``ts_rank(ts_corr(close, volume, 10), 20)`` 返回 20。窗口只取
    「跟在逗号或括号后的独立整数」；普通整数常量（如 1、2）不算窗口，
    因为要求前面是逗号/括号且公式含 ts_ 类算子的启发近似。窗口数值取
    公式中最大数字（与 §46.4 fast/medium/slow 分桶一致）。
    """
    text = str(formula)
    windows = [
        int(m.group(1))
        for m in _WINDOW_RE.finditer(text)
        if 3 <= int(m.group(1)) <= 10000
    ]
    return max(windows) if windows else 0


def horizon_bucket_of(formula: str) -> str:
    """§46.4：最大窗口 <10 → fast；10-60 → medium；>60 → slow；无窗口 → unknown。"""
    w = max_window_of(formula)
    if w <= 0:
        return "unknown"
    if w < 10:
        return "fast"
    if w <= 60:
        return "medium"
    return "slow"


# ---------------------------------------------------------------------------
# §19.2 复杂度分级（与 fitness 一致）
# ---------------------------------------------------------------------------

def complexity_bucket_of(ast_nodes: int) -> str:
    """§19.2（与 fitness.compute_search_fitness 的 P_complexity 分级一致）。

    nodes 0-24 → small；25-40 → mid；41+ → heavy。
    """
    if ast_nodes <= 24:
        return "small"
    if ast_nodes <= 40:
        return "mid"
    return "heavy"


# ---------------------------------------------------------------------------
# §46.3 Mechanism / Schema DNA：启发标签（多标签）
# ---------------------------------------------------------------------------

#: 机制启发规则：(算子名/字段 token 子串, 机制标签, 说明)
MECHANISM_RULES: tuple[tuple[str, str, str], ...] = (
    ("ts_rank", "momentum_or_corr", "时序排序通常刻画动量/相对强度"),
    ("ts_corr", "momentum_or_corr", "相关结构多用于动量/联动刻画"),
    ("rank_corr", "momentum_or_corr", "截面相关"),
    ("return_volume_corr", "momentum_or_corr", "量价相关"),
    ("abs_return_volume_corr", "momentum_or_corr", "量价相关"),
    ("volume", "liquidity", "量"),
    ("amount", "liquidity", "成交额"),
    ("turn", "liquidity", "换手"),
    ("turnover", "liquidity", "换手"),
    ("vol", "volatility", "波动（含 StdDev/RealizedVol 等子串）"),
    ("std", "volatility", "标准差"),
    ("skew", "volatility", "偏度"),
    ("kurt", "volatility", "峰度"),
    ("pe", "valuation", "估值"),
    ("pb", "valuation", "估值"),
    ("pcf", "valuation", "估值"),
    ("ps", "valuation", "估值"),
    ("market_cap", "valuation", "市值"),
    ("_mv", "valuation", "市值"),
)

#: 机制 → 说明（§46.3 人工标签不用于因果，仅作分组维度）
MECHANISM_DESCRIPTIONS: dict[str, str] = {
    "momentum_or_corr": "动量/相关结构",
    "reversal": "反转（负号壳启发）",
    "liquidity": "流动性（量/额/换手）",
    "volatility": "波动率结构",
    "valuation": "估值/市值",
}

#: 负号壳识别：形如 "-(...)"、"(-(...))"、"(0 - (...))" 整体取负
_NEG_SHELL_RE = re.compile(
    r"(?:^|[\s(])-\s*\([^)]*(?:\([^)]*\)[^)]*)*\)|"
    r"0\s*-\s*\("
)


def _has_neg_shell(formula: str) -> bool:
    return bool(_NEG_SHELL_RE.search(str(formula)))


def _operator_names_set(formula: str) -> set[str]:
    return set(_extract_operator_names(str(formula)))


def _tokens(text: str) -> set[str]:
    """公式文本中的完整标识符 token（避免子串误判，如 volume 里的 vol）。"""
    return {m.group(0) for m in _FIELD_TOKEN_RE.finditer(str(text))}


def mechanisms_of(formula: str) -> list[str]:
    """§46.3：机制启发标签（多标签）。

    - 含 ts_rank / ts_corr / rank_corr / return_volume_corr → momentum_or_corr；
    - 含负号壳（整体取负）→ reversal；
    - 含 volume / amount / turn / turnover → liquidity；
    - 含 vol / std / skew / kurt → volatility；
    - 含 pe / pb / pcf / ps / market_cap → valuation。
    """
    text = str(formula)
    ops = _operator_names_set(text)
    lower_text = text.lower()
    mechs: list[str] = []

    if any(k in ops or k in lower_text for k in ("ts_rank", "ts_corr", "rank_corr")):
        mechs.append("momentum_or_corr")
    elif any(k in ops or k in lower_text for k in ("return_volume_corr", "abs_return_volume_corr")):
        mechs.append("momentum_or_corr")

    if _has_neg_shell(text):
        mechs.append("reversal")

    if any(k in lower_text for k in ("volume", "amount", "turnover", "turn")):
        mechs.append("liquidity")

    # volatility：避免 "volume" 的 "vol" 子串误判（用精确 token + 算子名）
    vol_ops = {"ts_std", "std", "ts_skew", "skew", "ts_kurt", "kurt",
               "realized_vol", "volatility", "stddev"}
    if any(t in ops for t in vol_ops) or "volatility" in _tokens(lower_text):
        mechs.append("volatility")

    if any(k in lower_text for k in ("pe", "pb", "pcf", "ps", "market_cap", "_mv")):
        mechs.append("valuation")

    return mechs


# ---------------------------------------------------------------------------
# §75.5 FactorDNA 提取
# ---------------------------------------------------------------------------

def _window_hint_count(formula: str) -> int:
    """粗略的节点规模代理：算子调用数 + 独立字段 token 数。"""
    text = str(formula)
    n_ops = len(_extract_operator_names(text))
    n_fields = len({m.group(0) for m in _FIELD_TOKEN_RE.finditer(text)})
    return n_ops + n_fields


def dna_from_formula(
    formula: str,
    canonical_inspect: dict[str, Any] | None = None,
) -> FactorDNA:
    """从公式文本提取 FactorDNA。

    Parameters
    ----------
    formula : str
        factor_engine DSL 公式文本（例如 ``ts_rank(ts_corr(close, volume, 10), 20)``）。
    canonical_inspect : dict, optional
        可选的结构信息（来自 canonical 检查器）：
        - ``ast_nodes``：AST 节点数（缺省用公式规模代理）；
        - ``max_window``：最大窗口（缺省从公式数字启发）；
        - ``fields``：字段名列表（缺省从公式抽取）。
        契约：``FactorDNA`` 的 ``schema`` 字段存放原始输入快照。

    Returns
    -------
    FactorDNA
    """
    text = str(formula)
    info = dict(canonical_inspect or {})

    fields_raw: list[str] = list(info.get("fields") or []) or [
        m.group(0) for m in _FIELD_TOKEN_RE.finditer(text)
    ]
    families = list(info.get("field_families") or []) or field_families_of(text)

    ops: list[str]
    if "operators" in info:
        ops = list(info["operators"])
    else:
        # allowlist 只依赖 factor_engine 注册表，进程内恒定：memoize 一次，
        # 避免 3000 样本的 stats 每条公式都重算（34s → 毫秒）。
        _ALLOWLIST_CACHE = getattr(dna_from_formula, "_allowlist_cache", None)
        if _ALLOWLIST_CACHE is None:
            from functools import lru_cache

            @lru_cache(maxsize=1)
            def _allowlist():
                return default_operator_allowlist()

            _ALLOWLIST_CACHE = _allowlist
            dna_from_formula._allowlist_cache = _ALLOWLIST_CACHE
        allow = _ALLOWLIST_CACHE()
        counter = extract_operator_multiset(text, allowlist=allow)
        ops = sorted(counter.elements())

    # 去重保序（multiset 降为集合作为 FactorDNA.operators）
    seen: set[str] = set()
    unique_ops: list[str] = []
    for o in ops:
        if o not in seen:
            seen.add(o)
            unique_ops.append(o)

    w = int(info.get("max_window") or 0) or max_window_of(text)
    horizon = horizon_bucket_of(text) if not (info.get("horizon_bucket")) else info["horizon_bucket"]
    if "max_window" in info or "horizon_bucket" not in info:
        horizon = horizon_bucket_of(text)

    nodes = int(info.get("ast_nodes") or 0) or _window_hint_count(text)
    complexity = complexity_bucket_of(nodes)

    return FactorDNA(
        field_families=families,
        operators=unique_ops,
        ast_motifs=list(info.get("ast_motifs") or []),
        mechanisms=list(info.get("mechanisms") or []) or mechanisms_of(text),
        schema=info,
        horizon_bucket=horizon,
        complexity_bucket=complexity,
    )


# 兼容别名：complexity_bucket_of 的数值版（§19.2 阈值）
COMPLEXITY_BUCKET = complexity_bucket_of
