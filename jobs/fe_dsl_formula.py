"""
FE DSL 公式规范化器 (FE DSL formula normalizer)
==============================================

职责：
  1. 把 factor_delivery_converted 里那些杂乱（甚至错误）的 fe_formula
     转成 factor_engine 真能 parse 的 DSL 表达式；
  2. 如果表达不了，明确打标，让上层在 HTML 里渲染成
     "⚠ FE DSL 暂不支持 — 已保留原文" 的可视提示，避免出现 "（公式为空）"。

输入：单个 factor 的 JSON dict（含 code/formula/fe_formula/lqtp_formula/required_columns）。
输出：dict(fe_dsl, status, note) 其中
      status ∈ {"ok", "fallback", "empty"}
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

PROJECT = Path("/home/sunhaiwei/quant_projects")
FE_DIR = PROJECT / "factor_engine"
sys.path.insert(0, str(FE_DIR))
sys.path.insert(0, str(PROJECT / "factor_delivery_converted"))

try:
    from api.dsl_parser import parse_expr  # noqa
    _HAS_FE_PARSER = True
except Exception:  # pragma: no cover
    _HAS_FE_PARSER = False

try:
    import convert_factors as _cf_mod
except Exception:
    _cf_mod = None


# ---------- 1. 语法清理 ----------
_KW_FUNC_RE = re.compile(
    r"\b(ema|ts_mean|ts_std|ts_sum|ts_max|ts_min|ts_median|ts_rank|ts_delta|ts_pct|"
    r"ts_corr|ts_cov|ts_decay_linear|ts_decay_exp|ts_skew|ts_kurt|ts_argmax|ts_argmin|"
    r"ts_quantile|ts_regression_slope|ts_topk_sum|rolling_mean|rolling_std|rolling_max|"
    r"rolling_min|rolling_sum|rolling_median)\s*\(")


_KEYWORD_ARGS = {"span", "window", "period", "halflife", "halflife_", "alpha"}


def _flip_ema_kwarg(text: str) -> str:
    """把形如 ``ema(span=N, x)`` 翻转成 ``ema(x, span=N)``。

    先递归处理最深层的嵌套调用（从右往左），保证内层翻转完外层再处理。"""
    import ast

    def find_matching_close(s: str, start: int) -> int:
        depth = 0
        i = start
        while i < len(s):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    # 递归替换；先扫到最里层
    def process(s: str) -> str:
        # 先找里层：扫描所有 NAME(...)，对每一段递归处理，再判断当前调用是否要翻转
        i = 0
        while i < len(s):
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", s[i:])
            if not m:
                i += 1
                continue
            open_paren = i + m.end() - 1
            close = find_matching_close(s, open_paren)
            if close == -1:
                break
            inside = s[open_paren+1:close]
            # 递归先处理 inside
            new_inside = process(inside)
            # 看 new_inside 是不是以 KW=数字, ... 开头
            m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(.+)$",
                          new_inside)
            if m2 and m2.group(1) in _KEYWORD_ARGS and m2.group(3).strip():
                kw_name = m2.group(1)
                kw_val = m2.group(2)
                rest = m2.group(3)
                new_inside = f"{rest}, {kw_name}={kw_val}"
            s = s[:open_paren+1] + new_inside + s[close:]
            i = close + 1  # 跳过刚处理完的这一段，避免重扫
        return s

    return process(text)


def _strip_pseudo_assignments(text: str) -> str:
    """剥掉 x = ..., y = ... 的前缀 (FE DSL 是单一表达式)"""
    if "=" not in text:
        return text
    m = list(re.finditer(r"\bfactor\s*=\s*", text))
    if m:
        return text[m[-1].end():]
    parts = re.split(r";\s*", text)
    last = parts[-1].strip()
    if "=" in last:
        last = last.split("=", 1)[1].strip()
    return last


def _clean_basic_syntax(text: str) -> str:
    t = text.strip()
    # 修复 `...)=N)` 尾巴: ema(span=15, ...)=15) -> ema(span=15, ...)
    t = re.sub(r"\)\s*=\s*-?\d+(\.\d+)?\s*\)?$", ")", t).strip()
    # 行尾多余 )
    while t.count(")") > t.count("(") and t.endswith(")"):
        t = t[:-1].rstrip()
    # 大写 -> 小写 (注册算子)
    t = re.sub(r"\bEMA\b\s*\(", "ema(", t)
    t = re.sub(r"\bSMA\b\s*\(", "ts_mean(", t)
    t = re.sub(r"\bMA\b\s*\(", "ts_mean(", t)
    t = re.sub(r"\bSTD\b\s*\(", "ts_std(", t)
    t = re.sub(r"\bATR\b\s*\(", "ts_atr(", t)
    # bracket math noise
    t = re.sub(r"\[\|([^\[\]]+?)\|\]", r"abs(\1)", t)
    t = re.sub(r"\|([^|\[\]]+?)\|", r"abs(\1)", t)
    # subscript C_{t-k}
    t = re.sub(r"C_\{t-(\d+)\}", r"close.shift(\1)", t)
    t = re.sub(r"C_\{t\}", "close", t)
    t = re.sub(r"\bC_t\b", "close", t)
    # shift -> ts_delay
    t = re.sub(r"\bshift\s*\(", "ts_delay(", t)
    # if X then Y else Z -> where(X, Y, Z)
    t = re.sub(r"\bif\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)$",
               r"where(\1, \2, \3)", t)
    # 条件缩简: rolling(N).mean() -> ts_mean(N, x)
    t = re.sub(r"rolling\((\d+)\)\.mean\(\)", r"ts_mean(\1, x)", t)
    # EMA_volume(N) -> ema(N, volume)
    t = re.sub(r"EMA_volume\s*\(\s*(\d+)\s*\)", r"ema(\1, volume)", t)
    # MA_volume(N) -> ts_mean(N, volume)
    t = re.sub(r"MA_volume\s*\(\s*(\d+)\s*\)", r"ts_mean(\1, volume)", t)
    return t


def _try_parse(text: str) -> bool:
    if not text or not _HAS_FE_PARSER:
        return False
    try:
        parse_expr(text)
        return True
    except Exception:
        return False


# ---------- 2. 主体：从 factor JSON 推出最好的 FE DSL 表达式 ----------


def derive_fe_dsl(factor_json: dict) -> dict:
    """返回 {"dsl": str, "status": "ok|fallback", "note": str}"""
    code = (factor_json.get("code") or "").strip()
    src_formula = (factor_json.get("formula") or "").strip()
    fe_formula = (factor_json.get("fe_formula") or "").strip()
    lqtp_formula = (factor_json.get("lqtp_formula") or "").strip()
    rationale = (factor_json.get("rationale") or "").strip()

    candidates = []
    # 1) 已有 fe_formula, 清理
    if fe_formula:
        cand = _strip_pseudo_assignments(fe_formula)
        cand = _clean_basic_syntax(cand)
        cand = _flip_ema_kwarg(cand)
        candidates.append(("fe_formula", cand))

    # 2) 从 src_formula 重生成
    if src_formula and _cf_mod is not None:
        regenerated, _ = _cf_mod._convert_formula(src_formula, "fe")
        regenerated = _clean_basic_syntax(_strip_pseudo_assignments(regenerated))
        regenerated = _flip_ema_kwarg(regenerated)
        if (not fe_formula) or regenerated != candidates[-1][1] if candidates else True:
            candidates.append(("regenerated", regenerated))

    # 3) lqtp_formula 兜底
    if lqtp_formula:
        cand = _clean_basic_syntax(_strip_pseudo_assignments(lqtp_formula))
        cand = _flip_ema_kwarg(cand)
        candidates.append(("lqtp", cand))

    # 选第一个能 parse 的
    for tag, cand in candidates:
        if _try_parse(cand):
            return {"dsl": cand, "status": "ok",
                    "note": f"FE DSL 来源: {tag}"}

    return {
        "dsl": "",
        "status": "fallback",
        "note": ("⚠ FE DSL 暂不支持 — 因子使用了 factor_engine 表达不了的算子"
                 "（含自定义工具、复合多步算式、不可解析语法）。"
                 "已保留原始公式文本 + rationale 供阅读。"),
    }


# ---------- 3. 显示层 ----------


def _code_to_readable(code: str, max_lines: int = 12) -> str:
    """从 Python code 抽取 'return ...' 表达式用于回填空 formula"""
    if not code:
        return ""
    # 取 return 语句
    lines = []
    in_return = False
    cur = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("return "):
            in_return = True
            cur.append(s[len("return "):].rstrip(","))
            # 一行 return 一般就够
            lines.append("return " + cur[0])
            break
    # 也保留中间的计算
    inner_lines = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("#") or s.startswith("def "):
            continue
        if s.startswith("return "):
            break
        if "=" in s and not s.startswith(("==", ">", "<", "!=")):
            inner_lines.append(s.rstrip(","))
        if len(inner_lines) > max_lines - 1:
            break
    if inner_lines:
        inner_lines.append("return ..." if not lines else "")
    return "\n".join(inner_lines + lines).strip()


def display_formula_text(dsl: str, status: str, is_flipped: bool,
                          note: str, raw_formula: str, rationale: str,
                          code: str = "") -> dict:
    """返回 {"display": str, "explanation": str}"""
    parts = []
    if is_flipped:
        parts.append("−")
    if status == "ok" and dsl:
        parts.append(dsl)
    elif status == "fallback":
        # 三级回退: 原 formula 文本 → code 关键行 → rationale
        body = raw_formula
        if not body:
            body = _code_to_readable(code)
        if not body:
            body = rationale[:240]
        if not body:
            body = "（公式为空）"
        parts.append(f"（FE DSL 暂不支持 — 原文）{body}")
    else:
        parts.append("（公式为空）")

    display = "".join(parts)

    explanation = ""
    if status == "fallback":
        explanation = note
    elif is_flipped and status == "ok":
        explanation = ("原始 IC &lt; 0；展示公式已取负号，"
                       "页面上所有 LS / Sharpe / 年化 / 回撤等数值也已统一翻为正向。")
    return {"display": display, "explanation": explanation, "status": status}


if __name__ == "__main__":
    # self-test
    sample = {"fe_formula": "factor = ema(span=15, (close/shift(close,5)-1) * log(vol_ratio))=15)"}
    print(derive_fe_dsl(sample))
    sample = {"fe_formula": "", "formula": "raw = (high - low) * (volume / EMA_volume(20)); smoothed with EMA(10); then scale by 1.2 when resvol_high > 0, else 0.8; then cs_rank"}
    print(derive_fe_dsl(sample))
