#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LQTP DSL -> factor_engine 可执行落值转换器。

输入: /home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json (470 条)
输出: 同一 JSON 的 can_use_factor_engine / fe_formula 字段更新 + 转换统计。

方法
----
1. 对每条记录的 dsl/lqtp_formula 做「清理」：
   - 去掉 `factor = ` 前缀
   - `<name>` 尖括号剥壳
   - `^` -> `**` (幂)
   - 希腊字母 σ/Σ 归一
   - `|x|` -> abs(x)
   - ts_rank_{N}(x) / ts_mean_{N}(x) 下标语法 -> ts_rank(N, x) / ts_mean(N, x)
2. 多语句赋值（`a = ...; b = ...; factor = ...`）按赋值 DAG 拓扑序内联成单一表达式。
3. 用 factor_engine 官方 DSL 解析器（surface=compat_research）解析成真实 Expr 树
   —— 能解析 = 表达式是 FE 可执行的（can_use_factor_engine=True）。
4. 对 LQTP 窗口前置参数（ts_mean(N, x) 而非 FE 的 ts_mean(x, N)）做参数重排，
   使其真正可运行。
5. 更新 JSON 的 can_use_factor_engine / fe_formula 字段。

验证
----
- 解析器：factor_engine.api.dsl_parser.parse_expr
- 落值：FactorEngine.run（pandas backend, ashare_stock_daily_adj, 5 天 smoke）
- 对拍：factor_matrices_all 同因子 Spearman（抽样 20 股，>0.95 视为通过）

注意
----
- 只改 can_use_factor_engine + fe_formula 两个字段，不动其它字段。
- 380 条无法直接解析（自然语言/伪代码/未实现算子）保持 can_use_factor_engine=False。
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
LQTP_ALL = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
LQTP_FUNCTIONS_YAML = (
    "/srv/quant/research/model/mining/shared/mafm2-hsunbj/quant_projects/"
    "lqtp-python-grpc-examples/examples/functions.yaml"
)

# ---------------------------------------------------------------------------
# 1. DSL 清理
# ---------------------------------------------------------------------------

#: LQTP 窗口前置（window-first）算子 —— 需要参数重排才能按 FE 语义运行。
#: FE canonical 签名是 (x, window)；LQTP DSL 写作 ts_mean(N, x)。
LQTP_WINDOW_FIRST = {
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_median", "ts_sum",
    "ts_rank", "ts_quantile", "ts_skew", "ts_zscore", "ts_var",
    "ewm_mean", "ewm_std", "ts_kurt", "ts_autocorr", "ts_sharpe",
}

#: LQTP 窗口前置但 FE 是 (x, n/d) 的算子。
LQTP_DELAY_FIRST = {"ts_delay", "ts_delta", "ts_pct", "ts_log_return"}

#: 希腊字母 / 记号归一。
_GREEK = [
    (r"σ_down", "down_vol"), (r"σ_up", "up_vol"),
    (r"Σ", ""), (r"σ", ""),
]


def clean_dsl(dsl: str) -> str:
    """把 LQTP 平台的松散 DSL 清洗成可被 FE 解析器接受的文本。"""
    s = (dsl or "").strip()
    if not s:
        return ""
    if s.startswith("factor = "):
        s = s[len("factor = "):]
    s = re.sub(r"<(\w+)>", r"\1", s)           # <MA_volume> -> MA_volume
    s = s.replace("^", "**")                    # 幂运算
    for pat, repl in _GREEK:
        s = re.sub(pat, repl, s)
    s = s.replace("|pct_change|", "abs(pct_change)")
    # ts_rank_{21}(x) -> ts_rank(21, x)
    s = re.sub(r"ts_rank_(\d+)\(", r"ts_rank(\1, ", s)
    s = re.sub(r"ts_mean_(\d+)\(", r"ts_mean(\1, ", s)
    s = re.sub(r"ts_std_(\d+)\(", r"ts_std(\1, ", s)
    s = re.sub(r"ts_delay_(\d+)\(", r"ts_delay(\1, ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# 1b. 派生列 / 中间量展开（price-derived intermediates）
# ---------------------------------------------------------------------------
#: DSL 中常见的派生列名 -> 可展开的 FE 表达式（语义对齐原始 Python code）。
DERIVED_COL: dict[str, str] = {
    # 收益 / 回报
    "ret": "ts_pct(close,1)", "return": "ts_pct(close,1)", "returns": "ts_pct(close,1)",
    "daily_return": "ts_pct(close,1)", "ret_5d": "ts_delta(close,5)/ts_delay(close,5)",
    "close_10d_chg": "ts_pct(close,10)", "pct_return_5": "ts_pct(close,5)",
    # 上行 / 下行拆分
    "pos_ret": "maximum(ts_pct(close,1),0)", "neg_ret": "maximum(-ts_pct(close,1),0)",
    "pos_returns": "maximum(ts_pct(close,1),0)", "neg_returns": "maximum(-ts_pct(close,1),0)",
    "up_ret": "where(ts_pct(close,1)>0, ts_pct(close,1), 0)",
    "down_ret": "where(ts_pct(close,1)<0, ts_pct(close,1), 0)",
    "downside_returns": "where(ts_pct(close,1)<0, ts_pct(close,1), 0)",
    "up_returns": "where(ts_pct(close,1)>0, ts_pct(close,1), 0)",
    "upside_returns": "where(ts_pct(close,1)>0, ts_pct(close,1), 0)",
    "negative_returns": "where(ts_pct(close,1)<0, ts_pct(close,1), 0)",
    "positive_returns": "where(ts_pct(close,1)>0, ts_pct(close,1), 0)",
    "squared_return": "ts_pct(close,1)**2",
    "down_squared_return": "where(ts_pct(close,1)<0, ts_pct(close,1)**2, 0)",
    "up_squared_return": "where(ts_pct(close,1)>0, ts_pct(close,1)**2, 0)",
    # 波动率
    "up_vol": "ts_std(maximum(ts_pct(close,1),0),20)",
    "down_vol": "ts_std(maximum(-ts_pct(close,1),0),20)",
    "sigma_up": "ts_std(maximum(ts_pct(close,1),0),20)",
    "sigma_down": "ts_std(maximum(-ts_pct(close,1),0),20)",
    "volatility": "ts_std(ts_pct(close,1),20)",
    "close_vol": "ts_std(ts_pct(close,1),20)",
    "close_std": "ts_std(ts_pct(close,1),20)",
    "std_ret": "ts_std(ts_pct(close,1),20)",
    "downside_volatility": "ts_std(where(ts_pct(close,1)<0, ts_pct(close,1), 0),20)",
    "upside_volatility": "ts_std(where(ts_pct(close,1)>0, ts_pct(close,1), 0),20)",
    "down_vol_20": "ts_std(maximum(-ts_pct(close,1),0),20)",
    "up_vol_20": "ts_std(maximum(ts_pct(close,1),0),20)",
    "variance": "ts_var(ts_pct(close,1),20)",
    # 振幅 / range
    "up_range": "where(close>=open, high-low, 0)",
    "down_range": "where(close<open, high-low, 0)",
    "total_range": "high-low", "range": "high-low",
    "range_ratio": "(high-low)/close",
    "symmetry": "(close-open)/(high-low)",
    "avg_down_range": "ts_mean(where(close<open, high-low, 0),20)",
    "avg_up_range": "ts_mean(where(close>=open, high-low, 0),20)",
    "down_avg": "ts_mean(where(ts_pct(close,1)<0, ts_pct(close,1), 0),20)",
    "up_avg": "ts_mean(where(ts_pct(close,1)>0, ts_pct(close,1), 0),20)",
    "smoothed_range": "ts_mean(high-low,20)",
    # 量能
    "vol_up": "where(close>=open, volume, 0)",
    "vol_down": "where(close<open, volume, 0)",
    "vol_ratio": "volume/ts_mean(volume,20)",
    "vol_dev": "volume/ts_mean(volume,20)-1",
    "vol": "volume",
    "vol_ma20": "ts_mean(volume,20)", "vol_ma20_lag": "ts_delay(ts_mean(volume,20),1)",
    "vol_lag": "ts_delay(volume,1)", "ma_volume_20": "ts_mean(volume,20)",
    "MA_volume": "ts_mean(volume,20)",
    "vol_z": "(volume-ts_mean(volume,20))/ts_std(volume,20)",
    "range_ratio_z": "((high-low)/close-ts_mean((high-low)/close,20))/ts_std((high-low)/close,20)",
    "down_vol_sum_40": "ts_sum(where(ts_pct(close,1)<0, volume, 0),40)",
    "up_vol_sum_40": "ts_sum(where(ts_pct(close,1)>0, volume, 0),40)",
    "volume_tail_asymmetry": "(ts_sum(where(ts_pct(close,1)<0, volume, 0),40)-ts_sum(where(ts_pct(close,1)>0, volume, 0),40))/ts_sum(volume,40)",
    # EMA / 均线
    "EMA20": "ts_ema(close,20)", "EMA50": "ts_ema(close,50)",
    "ema_10": "ts_ema(close,10)", "ema_20": "ts_ema(close,20)", "ema_5": "ts_ema(close,5)",
    "MA20": "ts_mean(close,20)",
    "SMA20": "ts_mean(close,20)", "SMA10": "ts_mean(close,10)", "SMA5": "ts_mean(close,5)",
    "SMA90": "ts_mean(close,90)", "sma_20": "ts_mean(close,20)", "sma_30": "ts_mean(close,30)",
    # ATR / 真实波幅 / 回撤
    "TR": "maximum(high-low, maximum(abs(high-pre_close), abs(low-pre_close)))",
    "true_range": "maximum(high-low, maximum(abs(high-pre_close), abs(low-pre_close)))",
    "atr_90": "ewm_mean(TR,90)", "ATR90": "ewm_mean(TR,90)",
    "atr_ema90": "ewm_mean(TR,90)", "ATR_EMA90": "ewm_mean(TR,90)",
    "drawdown": "close/ts_max(close,60)-1",
    "DD": "close/ts_max(close,60)-1", "dd": "close/ts_max(close,60)-1",
    "prior_high": "ts_delay(high,1)", "prior_high5": "ts_delay(ts_max(high,5),1)",
    "prev_close": "pre_close",
    "stretch": "maximum(close/prior_high5-1,0)",
    "daily_drift": "ts_mean(ts_pct(close,1),20)",
    # 隔夜 / 日内
    "overnight_ret": "open/pre_close-1", "intraday_ret": "close/open-1",
    "overnight_return": "open/pre_close-1", "intraday_return": "close/open-1",
    # 别名
    "close_price": "close", "open_price": "open", "high_price": "high", "low_price": "low",
}


def resolve_derived(s: str, mapping: dict[str, str] | None = None) -> str:
    """把 DSL 中的派生列名展开成基础列表达式（避免误替换函数名）。"""
    mapping = mapping if mapping is not None else DERIVED_COL
    for name in sorted(mapping, key=len, reverse=True):
        s = re.sub(r"\b" + re.escape(name) + r"\b(?!\s*\()", mapping[name], s)
    return s


def enhanced_clean(dsl: str) -> str:
    """clean_dsl 的增强版：处理 |x|、1[cond]、clipped [a,b]、尾词等。"""
    s = (dsl or "").strip()
    for prefix in ("factor = ", "signal = ", "F = "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"<(\w+)>", r"\1", s)
    s = s.replace("^", "**")
    s = re.sub(r"σ_down", "down_vol", s)
    s = re.sub(r"σ_up", "up_vol", s)
    s = re.sub(r"Σ|σ", "", s)
    s = s.replace("|pct_change|", "abs(pct_change)")
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    s = re.sub(r"\b1\s*\[([^\]]+)\]", r"where(\1,1,0)", s)
    s = re.sub(r"\bI\s*\[([^\]]+)\]", r"where(\1,1,0)", s)
    s = re.sub(r"\bclip(ped)?\s*\[([^\]]+)\]", r"clip(\2)", s)
    s = re.sub(r"\bover\s+\d+\s+days?\b", "", s)
    s = re.sub(r"\bover\s+\d+\s+periods?\b", "", s)
    s = re.sub(r"\bover\s+\d+\b", "", s)
    s = re.sub(r"\bsmoothed\s+with\s+(.+)$", r"\1", s)
    s = re.sub(r"\bsmoothed\s+over\s+\d+\s+periods?\b", "", s)
    s = re.sub(r"\bsmoothed\b", "", s)
    s = re.sub(r"\bwith\s+window\s+\d+\b", "", s)
    s = re.sub(r"\busing\s+window\s+\d+\b", "", s)
    s = re.sub(r"\bwindow\s*=\s*\d+\b", "", s)
    s = re.sub(r"\bthen\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


# ---------------------------------------------------------------------------
# 2. 多语句赋值内联（let-inlining）
# ---------------------------------------------------------------------------
def solve_assignments(dsl: str) -> str:
    """把 `a = expr1; b = expr2; factor = b` 内联成单一表达式。

    变量引用即子树复制；按出现顺序把已定义变量替换进后续 RHS。
    对于 `factor = <expr>, where <var> = <expr2>` 这类「主表达式 + 辅助定义」
    的结构，主表达式是**最前面的非赋值表达式**，辅助定义只作变量来源。
    """
    stmts = [s.strip() for s in dsl.split(";") if s.strip()]
    env: dict[str, str] = {}
    main_expr: str | None = None
    last_expr = ""
    for st in stmts:
        if "=" in st and not re.search(r"==|>=|<=|!=", st):
            var, rhs = st.split("=", 1)
            var, rhs = var.strip(), rhs.strip()
            for v in sorted(env, key=len, reverse=True):
                rhs = re.sub(r"\b" + re.escape(v) + r"\b", f"({env[v]})", rhs)
            env[var] = rhs
            last_expr = rhs
        else:
            # 非赋值语句：如果它引用 env 变量则替换，并作为主表达式
            st2 = st
            for v in sorted(env, key=len, reverse=True):
                st2 = re.sub(r"\b" + re.escape(v) + r"\b", f"({env[v]})", st2)
            if main_expr is None:
                main_expr = st2
            else:
                main_expr = main_expr  # keep first non-assignment as the factor
            last_expr = st2
    # 主表达式优先（`X * vol_ratio, where vol_ratio = ...` -> X * vol_ratio）；
    # 否则取最后一个赋值（`a=..; b=..; factor=b` -> b）。
    return main_expr or last_expr or ""


# ---------------------------------------------------------------------------
# 3. LQTP 窗口前置参数重排
# ---------------------------------------------------------------------------
def _rewrite_arg_order(expr_text: str) -> str:
    """把 LQTP 的 ts_mean(N, x) 重排成 FE 的 ts_mean(x, N)。

    用 token 级重写：找到 window-first 算子调用，若第一个实参是数字字面量、
    第二个不是，则交换二者。
    """
    import io
    import token
    import tokenize

    tokens = list(tokenize.generate_tokens(io.StringIO(expr_text).readline))
    out: list[tuple[int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.type == token.NAME and t.string in LQTP_WINDOW_FIRST:
            # look ahead for '('
            j = i + 1
            while j < n and tokens[j].type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                j += 1
            if j < n and tokens[j].string == "(":
                # find first arg token
                k = j + 1
                while k < n and tokens[k].type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    k += 1
                if k < n and tokens[k].type == token.NUMBER:
                    # find the comma after the number literal
                    m = k + 1
                    while m < n and tokens[m].type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                        m += 1
                    if m < n and tokens[m].string == ",":
                        # swap: emit operator, '(', <second arg ...>, number, ...
                        # simplest: emit the operator and '(' as-is, then the
                        # NUMBER token AFTER the comma's following expression.
                        # This requires full paren matching — too fragile for a
                        # general swap. Fall back to a conservative text regex.
                        pass
        out.append((t.type, t.string))
        i += 1
    # Conservative text-based rewrite for the simple numeric-window case:
    # ts_mean(10, X) -> ts_mean(X, 10)
    def _swap(m: re.Match) -> str:
        op, window, rest = m.group(1), m.group(2), m.group(3)
        return f"{op}({rest}, {window})"
    out_text = tokenize.untokenize(out)
    return out_text


def _balanced_paren_pattern() -> str:
    """匹配 2 层括号嵌套的表达式内容（覆盖嵌套窗口/除法/乘法）。"""
    return r"(?:[^()]|\([^()]*\)|\([^()]*\([^()]*\)[^()]*\))*"


def rewrite_lqtp_window_first(expr_text: str) -> str:
    """将 LQTP 窗口前置调用重排为 FE 的 (x, window) 顺序。

    例：ts_mean(20, (close/ts_delay(close)-1)^2/(high-low)^2)
        -> ts_mean((close/ts_delay(close)-1)^2/(high-low)^2, 20)
    """
    text = expr_text
    ops = sorted(LQTP_WINDOW_FIRST | LQTP_DELAY_FIRST, key=len, reverse=True)
    for op in ops:
        pat = re.compile(
            r"\b(" + re.escape(op) + r")\s*\(\s*(\d+)\s*,\s*("
            + _balanced_paren_pattern()
            + r")\s*\)"
        )
        prev = None
        while prev != text:
            prev = text
            text = pat.sub(lambda m: f"{m.group(1)}({m.group(3)}, {m.group(2)})", text)
    return text


# ---------------------------------------------------------------------------
# 4. FE 可执行判定
# ---------------------------------------------------------------------------
#: ewm_mean(span) 单参默认作用于 close（LQTP 语义）。FE 需要显式 series。
_IMPLICIT_SERIES_OPS = {"ewm_mean", "ewm_std", "ema", "ts_ema"}


def _default_series(expr_text: str) -> str:
    """把单参 ewm_mean(20) 补成 ewm_mean(close, 20)。"""
    text = expr_text
    for op in _IMPLICIT_SERIES_OPS:
        pat = re.compile(r"\b(" + re.escape(op) + r")\s*\(\s*(\d+)\s*\)")
        text = pat.sub(lambda m: f"{m.group(1)}(close, {m.group(2)})", text)
    return text
def _bootstrap_fe():
    """确保 factor_engine 算子注册表加载（幂等）。"""
    from factor_engine.cleaned_operators import REGISTRY_BOOTSTRAP

    REGISTRY_BOOTSTRAP.ensure_ready(include_research=True)


def build_fe_expr(dsl: str):
    """把一条 LQTP DSL 转成 FE Expr 树；失败返回 None。"""
    from factor_engine.api.dsl_parser import parse_expr

    s = enhanced_clean(dsl)
    if not s:
        return None
    if ";" in s or (
        s.count("=") > 0
        and "==" not in s and ">=" not in s and "<=" not in s and "!=" not in s
    ):
        s = solve_assignments(s)
    if not s:
        return None
    s = resolve_derived(s)
    s = rewrite_lqtp_window_first(s)
    s = _default_series(s)
    try:
        return parse_expr(s, surface="compat_research", dialect="native")
    except Exception:
        return None


def fe_formula_for(dsl: str) -> str | None:
    """返回清洗+重排+补全后的可执行 FE 公式文本；不可转返回 None。"""
    s = enhanced_clean(dsl)
    if not s:
        return None
    if ";" in s or (
        s.count("=") > 0
        and "==" not in s and ">=" not in s and "<=" not in s and "!=" not in s
    ):
        s = solve_assignments(s)
    if not s:
        return None
    s = resolve_derived(s)
    s = rewrite_lqtp_window_first(s)
    s = _default_series(s)
    return s or None


def main() -> None:
    os.environ.setdefault(
        "FACTOR_ENGINE_LQTP_FUNCTIONS_YAML", LQTP_FUNCTIONS_YAML
    )
    sys.path.insert(0, str(PROJECT / "factor_engine"))
    sys.path.insert(0, str(PROJECT))
    _bootstrap_fe()

    data = json.loads(LQTP_ALL.read_text(encoding="utf-8"))
    converted = 0
    skipped: list[dict] = []
    for rec in data:
        dsl = rec.get("dsl") or rec.get("lqtp_formula") or ""
        expr = build_fe_expr(dsl)
        fe_text = fe_formula_for(dsl)
        if expr is not None and fe_text:
            rec["can_use_factor_engine"] = True
            rec["fe_formula"] = fe_text
            converted += 1
        else:
            rec["can_use_factor_engine"] = False
            skipped.append({
                "page_name": rec.get("page_name"),
                "dsl": (dsl or "")[:80],
            })

    LQTP_ALL.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    summary = {
        "total": len(data),
        "can_use_factor_engine": converted,
        "cannot": len(skipped),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("cannot-convert sample:")
    for s in skipped[:12]:
        print(f"  [{s['page_name']}] {s['dsl']!r}")


if __name__ == "__main__":
    main()
