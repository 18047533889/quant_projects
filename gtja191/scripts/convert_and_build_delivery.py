#!/usr/bin/env python3
"""将 GTJA-191 原始公式转换为 factor_engine DSL，并生成 disk.v1 投递包。"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.data_source import campaign_data_source  # noqa: E402
from lib.dsl_normalize import normalize_operator_names  # noqa: E402
from lib.dsl_validate import validate_formula  # noqa: E402

GTJA_ROOT = PACKAGE_ROOT

# ---------------------------------------------------------------------------
# 手工 DSL（原始 GTJA 语法无法自动翻译或需简化）
# ---------------------------------------------------------------------------
MANUAL_DSL: dict[str, str] = {
    # REGBETA(..., SEQUENCE(n)) = 对时间序号回归 → ts_time_slope（compat）
    "gtja191_alpha_021": "ts_time_slope(ts_mean(close, 6), 6)",
    "gtja191_alpha_027": (
        'WMA((close - ts_delay(close, 3)) / (ts_delay(close, 3) + 1e-08) * 100 + (close - ts_delay(close, 6)) / (ts_delay(close, 6) + 1e-08) * 100, 12)'
    ),
    "gtja191_alpha_030": '0 * close',
    # 源式 corr 窗口=13、外层 ^5；勿写成 power(13) 进 corr 窗口
    "gtja191_alpha_056": (
        "rank(open - ts_min(open, 12)) < "
        "rank(power(rank(ts_corr(ts_sum((high + low) / 2, 19), "
        "ts_sum(ts_mean(volume, 40), 19), 13)), 5))"
    ),
    "gtja191_alpha_075": (
        'ts_sum(where(and_(close > open, index_close < index_open), 1, 0), 50) / (ts_sum(where(index_close < index_open, 1, 0), 50) + 1e-08)'
    ),
    "gtja191_alpha_116": "ts_time_slope(close, 20)",
    "gtja191_alpha_147": "ts_time_slope(ts_mean(close, 12), 12)",
    "gtja191_alpha_069": (
        'where(ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) > ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20), (ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) - ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20)) / (ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) + 1e-08), where(ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) == ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20), 0, (ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) - ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20)) / (ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 20) + 1e-08)))'
    ),
    "gtja191_alpha_143": 'where(close > ts_delay(close, 1), (close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08), 0)',
    "gtja191_alpha_149": (
        "ts_regression(where(index_close < ts_delay(index_close, 1), close / ts_delay(close, 1) - 1, 0), where(index_close < ts_delay(index_close, 1), index_close / ts_delay(index_close, 1) - 1, 0), 252, 0, 'slope')"
    ),
    "gtja191_alpha_181": (
        'ts_sum(close / ts_delay(close, 1) - 1 - ts_mean(close / ts_delay(close, 1) - 1, 20) - power(index_close - ts_mean(index_close, 20), 2), 20) / (ts_sum(power(index_close - ts_mean(index_close, 20), 3), 20) + 1e-08)'
    ),
    "gtja191_alpha_182": (
        'ts_sum(where(or_(and_(close > open, index_close > index_open), and_(close < open, index_close < index_open)), 1, 0), 20) / 20'
    ),
    "gtja191_alpha_183": '0 * close',
    "gtja191_alpha_186": (
        '(ts_mean(abs((ts_sum(where(and_(ts_delay(low, 1) - low > 0, ts_delay(low, 1) - low > high - ts_delay(high, 1)), ts_delay(low, 1) - low, 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) - ts_sum(where(and_(high - ts_delay(high, 1) > 0, high - ts_delay(high, 1) > ts_delay(low, 1) - low), high - ts_delay(high, 1), 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08)) / (ts_sum(where(and_(ts_delay(low, 1) - low > 0, ts_delay(low, 1) - low > high - ts_delay(high, 1)), ts_delay(low, 1) - low, 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) + ts_sum(where(and_(high - ts_delay(high, 1) > 0, high - ts_delay(high, 1) > ts_delay(low, 1) - low), high - ts_delay(high, 1), 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) + 1e-08) * 100), 6) + ts_delay(ts_mean(abs((ts_sum(where(and_(ts_delay(low, 1) - low > 0, ts_delay(low, 1) - low > high - ts_delay(high, 1)), ts_delay(low, 1) - low, 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) - ts_sum(where(and_(high - ts_delay(high, 1) > 0, high - ts_delay(high, 1) > ts_delay(low, 1) - low), high - ts_delay(high, 1), 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08)) / (ts_sum(where(and_(ts_delay(low, 1) - low > 0, ts_delay(low, 1) - low > high - ts_delay(high, 1)), ts_delay(low, 1) - low, 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) + ts_sum(where(and_(high - ts_delay(high, 1) > 0, high - ts_delay(high, 1) > ts_delay(low, 1) - low), high - ts_delay(high, 1), 0), 14) * 100 / (ts_sum(flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + 1e-08) + 1e-08) * 100), 6), 6)) / 2'
    ),
}


def _load_manual_dsl() -> dict[str, str]:
    merged = dict(MANUAL_DSL)
    override_path = GTJA_ROOT / "dsl" / "manual_dsl_overrides.json"
    if override_path.exists():
        merged.update(json.loads(override_path.read_text(encoding="utf-8")))
    return {name: _post_process_dsl(dsl) for name, dsl in merged.items()}


# 依赖 index_close/index_open，当前 PV 主表无 benchmark 列，暂不投递
# 030/183 为零因子 stub，不进入投递包
DELIVERY_EXCLUDED: frozenset[str] = frozenset({
    "gtja191_alpha_030",
    "gtja191_alpha_075",
    "gtja191_alpha_149",
    "gtja191_alpha_181",
    "gtja191_alpha_182",
    "gtja191_alpha_183",
})
ADX_HELPERS = {
    "hd": "high - ts_delay(high, 1)",
    "ld": "ts_delay(low, 1) - low",
    "tr": "flex_max(flex_max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1)))",
}

# vwap 在 allowlist 中同时是算子名，公式里用典型价量代理避免解析冲突
VWAP_PROXY = "((high + low + close) / 3)"

FIELD_MAP = {
    "CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "VOLUME": "volume",
    "VOL": "volume",
    "VWAP": VWAP_PROXY,
    "AMOUNT": "amount",
    "RET": "ret",
    "RETURN": "ret",
    "BANCHMARKINDEXCLOSE": "index_close",
    "BANCHMARKINDEXOPEN": "index_open",
    "HGIH": "high",
    "MKT": "mkt",
    "SMB": "smb",
    "HML": "hml",
}

OP_MAP = [
    (r"\bDECAYLINEAR\b", "ts_decay_linear"),
    (r"\bTS_CORR\b", "ts_corr"),
    (r"\bCORR\b", "ts_corr"),
    (r"\bCOVARIANCE\b", "ts_cov"),
    (r"\bCOVIANCE\b", "ts_cov"),
    (r"\bTS_COV\b", "ts_cov"),
    (r"\bTS_SUM\b", "ts_sum"),
    (r"\bSUMAC\b", "ts_sum"),
    (r"\bSUM\b", "ts_sum"),
    (r"\bTS_MEAN\b", "ts_mean"),
    (r"\bMEAN\b", "ts_mean"),
    (r"\bTS_STD\b", "ts_std"),
    (r"\bSTD\b", "ts_std"),
    (r"\bTS_MAX\b", "ts_max"),
    (r"\bTSMIN\b", "ts_min"),
    (r"\bTS_MIN\b", "ts_min"),
    (r"\bTSMAX\b", "ts_max"),
    (r"\bTSRANK\b", "ts_rank"),
    (r"\bMA\b", "ts_mean"),
    (r"\bDELAY\b", "ts_delay"),
    (r"\bDELTA\b", "ts_delta"),
    (r"\bDELAT\b", "ts_delta"),
    (r"\bRANK\b", "rank"),
    (r"\bABS\b", "abs"),
    (r"\bLOG\b", "log"),
    (r"\bSIGN\b", "sign"),
    (r"\bSCALE\b", "scale"),
    (r"\bMAX\b", "max"),
    (r"\bMIN\b", "min"),
    (r"\bCOUNT\b", "__COUNT__"),
    (r"\bPROD\b", "ts_product"),
    (r"\bSMEAN\b", "ts_ema"),
    (r"\bREGRESI\b", "ts_regression"),
    (r"\bFILTER\b", "__FILTER__"),
    (r"\bSELF\b", "close"),
    (r"\bSUMIF\b", "__SUMIF__"),
    (r"\bHIGHDAY\b", "ts_argmax"),
    (r"\bLOWDAY\b", "ts_argmin"),
    (r"\bSMA\b", "ts_ema"),
    (r"\bEMA\b", "ts_ema"),
    (r"\bWMA\b", "WMA"),
]


def _expand_adx_helpers(formula: str) -> str:
    if "gtja191_alpha_186" not in formula and "hd" not in formula:
        return formula
    prefix = " ".join(f"({expr})" if k in ("hd", "ld", "tr") else expr for k, expr in [])  # noqa: placeholder
    # 内联替换：在最终 DSL 里直接用 helper 定义
    out = formula
    for name, expr in ADX_HELPERS.items():
        out = out.replace(name, f"({expr})")
    return out


def _replace_power(expr: str) -> str:
    # 先处理括号表达式幂次
    pat_paren = re.compile(r"\(([^()]+)\)\s*\^\s*([0-9.]+)")
    prev = None
    cur = expr
    while prev != cur:
        prev = cur
        cur = pat_paren.sub(r"power((\1), \2)", cur)
    pat = re.compile(r"([\w\)\]]+)\s*\^\s*([0-9.]+)")
    prev = None
    while prev != cur:
        prev = cur
        cur = pat.sub(r"power(\1, \2)", cur)
    return cur


def _paren_depths(s: str) -> list[int]:
    depths: list[int] = []
    d = 0
    for ch in s:
        if ch == "(":
            d += 1
        depths.append(d)
        if ch == ")":
            d -= 1
    return depths


def _replace_ternary(expr: str) -> str:
    # 先修正 GTJA 单等号比较
    expr = re.sub(r"([a-z_]+)=([a-z_\(])", r"\1==\2", expr, flags=re.IGNORECASE)
    expr = expr.replace(".*", "*")
    while "?" in expr:
        depths = _paren_depths(expr)
        q_positions = [i for i, ch in enumerate(expr) if ch == "?"]
        if not q_positions:
            break
        # 选最深层 ? 中最靠右的一个
        max_depth = max(depths[i] for i in q_positions)
        q = max(i for i in q_positions if depths[i] == max_depth)
        target_depth = depths[q]
        colon = -1
        depth = 0
        for j in range(q + 1, len(expr)):
            ch = expr[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == target_depth:
                colon = j
                break
        if colon < 0:
            break
        cond = expr[:q].strip()
        true_part = expr[q + 1 : colon].strip()
        false_part = expr[colon + 1 :].strip()
        expr = f"where({cond}, {true_part}, {false_part})"
    return expr


def _find_matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i].strip())
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(s[start:].strip())
    return parts


def _replace_if_calls(expr: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(expr):
        m = re.match(r"IF\s*\(", expr[i:], flags=re.IGNORECASE)
        if not m:
            out.append(expr[i])
            i += 1
            continue
        start = i + m.end() - 1
        end = _find_matching_paren(expr, start)
        if end < 0:
            out.append(expr[i])
            i += 1
            continue
        inner = expr[start + 1 : end]
        args = _split_top_level(inner, ",")
        if len(args) == 3:
            out.append(f"where({args[0]}, {args[1]}, {args[2]})")
        else:
            out.append(expr[i : end + 1])
        i = end + 1
    return "".join(out)


def _replace_logic(expr: str) -> str:
    expr = expr.replace("&&", " and_ ")
    expr = expr.replace("||", " or_ ")
    expr = re.sub(r"(?<![\w_])&(?![\w_])", " and_ ", expr)
    return expr


def _normalize_fields(expr: str) -> str:
    for src, dst in FIELD_MAP.items():
        expr = re.sub(rf"\b{src}\b", dst, expr, flags=re.IGNORECASE)
    return expr


def _normalize_ops(expr: str) -> str:
    for pat, repl in OP_MAP:
        expr = re.sub(pat, repl, expr, flags=re.IGNORECASE)
    return expr


def _gtja_sma_span(n: int, m: int) -> int:
    """GTJA SMA(x,n,m) 的 alpha=m/n，对应 EMA span ≈ 2n/m - 1。"""
    return max(1, int(round(2 * n / m - 1)))


def _fix_ema_three_arg(expr: str) -> str:
    """将 GTJA SMA(x,n,m) 遗留的 ts_ema 三参数调用折叠为 ts_ema(x, span)。"""
    changed = True
    token = "ts_ema("
    while changed:
        changed = False
        i = 0
        while True:
            pos = expr.find(token, i)
            if pos < 0:
                break
            start = pos + len("ts_ema")
            end = _find_matching_paren(expr, start)
            if end < 0:
                break
            inner = expr[start + 1 : end]
            args = _split_top_level(inner, ",")
            if len(args) == 3:
                x, n, m = args
                try:
                    n_i, m_i = int(float(n.strip())), int(float(m.strip()))
                    span = _gtja_sma_span(n_i, m_i)
                    expr = expr[:pos] + f"ts_ema({x}, {span})" + expr[end + 1 :]
                    changed = True
                    i = pos + 1
                    continue
                except ValueError:
                    pass
            i = end + 1
    return expr


_ROLLING_FIELDS = r"(?:close|open|high|low|volume|ret|amount)"


def _is_int_literal(s: str) -> int | None:
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    return None


def _fix_rolling_max_min_calls(expr: str) -> str:
    """GTJA MAX(x,n)/MIN(x,n) 中 n 为整数窗口时 → ts_max/ts_min（保留 max(0,…) 等逐元素写法）。"""
    for fn, ts_fn in (("max", "ts_max"), ("min", "ts_min")):
        changed = True
        prefix = f"{fn}("
        while changed:
            changed = False
            i = 0
            while True:
                pos = expr.find(prefix, i)
                if pos < 0:
                    break
                if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
                    i = pos + 1
                    continue
                start = pos + len(fn)
                end = _find_matching_paren(expr, start)
                if end < 0:
                    break
                inner = expr[start + 1 : end]
                args = _split_top_level(inner, ",")
                if len(args) == 2:
                    a1, a2 = args[0].strip(), args[1].strip()
                    n = _is_int_literal(a2)
                    if (
                        n is not None
                        and n >= 2
                        and a1 != "0"
                        and not re.fullmatch(r"-?\d+(\.\d+)?", a1)
                    ):
                        expr = expr[:pos] + f"{ts_fn}({args[0]}, {n})" + expr[end + 1 :]
                        changed = True
                i = pos + 1
    return expr


def _fix_rolling_max_min(expr: str) -> str:
    """单字段 + 整数窗口（遗留路径）+ 通用复合表达式滚动 max/min。"""
    expr = re.sub(
        rf"\bmax\(({_ROLLING_FIELDS})\s*,\s*(\d+)\s*\)",
        r"ts_max(\1, \2)",
        expr,
    )
    expr = re.sub(
        rf"\bmin\(({_ROLLING_FIELDS})\s*,\s*(\d+)\s*\)",
        r"ts_min(\1, \2)",
        expr,
    )
    return _fix_rolling_max_min_calls(expr)


def _post_process_dsl(expr: str) -> str:
    expr = _fix_ema_three_arg(expr)
    expr = _fix_rolling_max_min(expr)
    expr = normalize_operator_names(expr)
    return re.sub(r"\s+", " ", expr).strip()


def _fix_sma_ema_calls(expr: str) -> str:
    return _fix_ema_three_arg(expr)


def _split_top_level_args(s: str) -> list[str]:
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def _fix_sumif(expr: str) -> str:
    out: list[str] = []
    i = 0
    token = "__SUMIF__"
    while i < len(expr):
        pos = expr.find(f"{token}(", i)
        if pos < 0:
            out.append(expr[i:])
            break
        out.append(expr[i:pos])
        start = pos + len(token)
        end = _find_matching_paren(expr, start)
        if end < 0:
            out.append(expr[pos:])
            break
        inner = expr[start + 1 : end]
        args = _split_top_level(inner, ",")
        if len(args) == 3:
            val, window, cond = args
            out.append(f"ts_sum(where({cond}, {val}, 0), {window})")
        else:
            out.append(expr[pos : end + 1])
        i = end + 1
    return "".join(out)


def _fix_filter(expr: str) -> str:
    out: list[str] = []
    i = 0
    token = "__FILTER__"
    while i < len(expr):
        pos = expr.find(f"{token}(", i)
        if pos < 0:
            out.append(expr[i:])
            break
        out.append(expr[i:pos])
        start = pos + len(token)
        end = _find_matching_paren(expr, start)
        if end < 0:
            out.append(expr[pos:])
            break
        inner = expr[start + 1 : end]
        args = _split_top_level(inner, ",")
        if len(args) == 2:
            val, cond = args
            out.append(f"where({cond}, {val}, 0)")
        else:
            out.append(expr[pos : end + 1])
        i = end + 1
    return "".join(out)


def _fix_count(expr: str) -> str:
    out: list[str] = []
    i = 0
    token = "__COUNT__"
    while i < len(expr):
        pos = expr.find(f"{token}(", i)
        if pos < 0:
            out.append(expr[i:])
            break
        out.append(expr[i:pos])
        start = pos + len(token)
        end = _find_matching_paren(expr, start)
        if end < 0:
            out.append(expr[pos:])
            break
        inner = expr[start + 1 : end]
        args = _split_top_level(inner, ",")
        if len(args) == 2:
            out.append(f"ts_sum(where({args[0]}, 1, 0), {args[1]})")
        else:
            out.append(expr[pos : end + 1])
        i = end + 1
    return "".join(out)


def _fix_highday_lowday(expr: str) -> str:
    expr = re.sub(
        r"\(\s*(\d+)\s*-\s*ts_argmax\(([^,]+),\s*(\d+)\)\s*\)",
        r"(\1 - (\3 - ts_argmax(\2, \3)))",
        expr,
    )
    expr = re.sub(
        r"\(\s*(\d+)\s*-\s*ts_argmin\(([^,]+),\s*(\d+)\)\s*\)",
        r"(\1 - (\3 - ts_argmin(\2, \3)))",
        expr,
    )
    return expr


def _protect_divisions(expr: str) -> str:
    # 分母加 epsilon 的常见裸除法在 GTJA 已部分处理；保持原样
    return expr


def convert_gtja_to_dsl(raw: str) -> str:
    expr = raw.strip()
    expr = expr.replace("–", "-").replace("，", ",")
    expr = expr.replace("./", "/")
    expr = _normalize_fields(expr)
    expr = _replace_power(expr)
    expr = _replace_if_calls(expr)
    expr = _replace_ternary(expr)
    expr = _replace_logic(expr)
    expr = _normalize_ops(expr)
    expr = _fix_count(expr)
    expr = _fix_sumif(expr)
    expr = _fix_filter(expr)
    expr = _fix_highday_lowday(expr)
    expr = _post_process_dsl(expr)
    return expr


def _candidate_hash(formula: str, universe_id: str, frequency_bucket: str) -> str:
    normalized = " ".join(formula.split())
    payload = f"{normalized}|{universe_id}|{frequency_bucket}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _validate(formula: str) -> tuple[bool, str]:
    return validate_formula(formula)


def build_dsl_catalog(source_path: Path) -> dict[str, dict]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    catalog: dict[str, dict] = {}
    manual = _load_manual_dsl()
    for name, raw in sorted(source.items()):
        if name in manual:
            dsl = _post_process_dsl(manual[name])
            note = "manual_dsl"
        else:
            dsl = convert_gtja_to_dsl(raw)
            note = "converted"
        if name == "gtja191_alpha_186":
            for k, v in ADX_HELPERS.items():
                dsl = dsl.replace(k, f"({v})")
        ok, msg = _validate(dsl)
        excluded = name in DELIVERY_EXCLUDED
        if excluded and name in {"gtja191_alpha_030", "gtja191_alpha_183"}:
            exclude_reason = "zero_stub"
        elif excluded:
            exclude_reason = "needs_benchmark_index_close_open"
        else:
            exclude_reason = ""
        catalog[name] = {
            "factor_name": name,
            "source_formula": raw,
            "dsl_formula": dsl,
            "conversion": note,
            "valid": ok,
            "validation": msg,
            "delivery_excluded": excluded,
            "delivery_excluded_reason": exclude_reason,
        }
    return catalog


def write_formula_files(catalog: dict[str, dict], formulas_dir: Path) -> None:
    formulas_dir.mkdir(parents=True, exist_ok=True)
    for name, item in sorted(catalog.items()):
        alpha_num = int(name.rsplit("_", 1)[-1])
        path = formulas_dir / f"gtja191_alpha_{alpha_num:03d}.dsl"
        lines = [
            f"# GTJA-191 Alpha {alpha_num:03d}",
            f"# source: {item['source_formula']}",
            "",
            item["dsl_formula"],
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


def build_campaign(
    catalog: dict[str, dict],
    *,
    campaign_id: str,
    out_dir: Path,
) -> None:
    universe_id = "A_SHARE_ALL_A_EX_ST"
    frequency_bucket = "daily"
    born_base = datetime(2026, 7, 4, 16, 0, 0, tzinfo=timezone.utc)

    config = {
        "schema_version": "disk.v1",
        "campaign_id": campaign_id,
        "generator_name": "manual",
        "generator_version": "1.0.0",
        "market": "ashare",
        "universe_id": universe_id,
        "signal_structure": "cross_sectional",
        "asset_class": "equity",
        "frequency_bucket": frequency_bucket,
        "domain_root": "price_volume",
        "domain": "price_volume",
        "mining_scope": {
            "primary_tables": ["StockDailyBar", "StockCapitalDaily"],
            "auxiliary_tables": ["StockList", "StockStatus", "StockIndustry", "Calendar"],
            "forbidden_tables": ["StockBalance", "StockIncome", "StockCashFlow"],
        },
        "operator_policy": "lqtp_pv_daily",
        # AFV campaign 路径提示（与 week2 / FE miner_delivery_spec §2.8 一致）
        "data_source": campaign_data_source(),
        # 本包公式含 ts_ema / flex_max / ts_time_slope 等，须 surface=compat 解析
        "dsl_surface": "compat",
        "mined_by": "zhangborui",
        "created_at": "2026-07-04T16:00:00Z",
        "mining_config": {
            "train_period": ["2014-01-01", "2021-12-31"],
            "valid_period": ["2022-01-01", "2023-12-31"],
            "test_period": ["2024-01-01", "2026-06-25"],
        },
        "mining_run_stats": {
            "llm_model": "manual-gtja191-port",
            "llm_provider": "manual",
            "llm_base_url": "",
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "started_at": "2026-07-04T16:00:00Z",
            "finished_at": "2026-07-04T16:05:00Z",
            "duration_seconds": 300,
            "candidates_generated": sum(
                1
                for v in catalog.values()
                if v["valid"] and not v.get("delivery_excluded")
            ),
            "candidates_submitted": sum(
                1
                for v in catalog.values()
                if v["valid"] and not v.get("delivery_excluded")
            ),
        },
    }

    campaign_dir = out_dir / campaign_id
    if campaign_dir.exists():
        for stale in campaign_dir.rglob("manifest 2.json"):
            stale.unlink()
        for child in campaign_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    seq = 0
    for name, item in sorted(catalog.items()):
        if not item["valid"] or item.get("delivery_excluded"):
            continue
        seq += 1
        formula = item["dsl_formula"]
        h8 = _candidate_hash(formula, universe_id, frequency_bucket)
        ts = born_base.replace(second=min(seq, 59))
        candidate_id = f"manual_{ts.strftime('%Y%m%d%H%M%S')}_{h8}"
        alpha_num = int(name.rsplit("_", 1)[-1])
        manifest = {
            "schema_version": "disk.v1",
            "candidate_id": candidate_id,
            "campaign_id": campaign_id,
            "formula": formula,
            "expression_type": "dsl",
            "dsl_surface": "compat",
            "generator_name": "manual",
            "generator_version": "1.0.0",
            "mined_by": "zhangborui",
            "market": "ashare",
            "universe_id": universe_id,
            "domain_root": "price_volume",
            "domain": "price_volume",
            "frequency_bucket": frequency_bucket,
            "signal_structure": "cross_sectional",
            "asset_class": "equity",
            "born_timestamp": ts.isoformat().replace("+00:00", "Z"),
            "metrics": {
                "train": {"ic": None, "icir": None, "rank_ic": None},
                "valid": {"ic": None, "icir": None, "rank_ic": None},
                "test": {"ic": None, "icir": None, "rank_ic": None},
            },
            "description": (
                f"国泰君安 GTJA-191 第 {alpha_num} 号因子，由原始价量公式翻译为 factor_engine DSL。"
                f" 来源公式: {item['source_formula'][:200]}"
            ),
        }
        cand_dir = campaign_dir / candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    source_path = GTJA_ROOT / "source" / "gtja191_formulas.json"
    if not source_path.exists():
        from parse_gtja_source import main as parse_main  # type: ignore

        parse_main()

    catalog = build_dsl_catalog(source_path)
    dsl_out = GTJA_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    dsl_out.parent.mkdir(parents=True, exist_ok=True)
    dsl_out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_formula_files(catalog, GTJA_ROOT / "formulas")

    valid_n = sum(1 for v in catalog.values() if v["valid"])
    deliver_n = sum(
        1 for v in catalog.values() if v["valid"] and not v.get("delivery_excluded")
    )
    invalid = [k for k, v in catalog.items() if not v["valid"]]
    print(f"DSL catalog: {valid_n}/{len(catalog)} valid, {deliver_n} deliverable -> {dsl_out}")
    if invalid:
        print("invalid:", invalid[:20], "..." if len(invalid) > 20 else "")

    campaign_id = "manual_ashare_pv_202607041600"
    build_campaign(catalog, campaign_id=campaign_id, out_dir=GTJA_ROOT / "candidate_pool")
    print(f"campaign -> {GTJA_ROOT / 'candidate_pool' / campaign_id}")
    return 0 if valid_n == 191 else 1


if __name__ == "__main__":
    raise SystemExit(main())
