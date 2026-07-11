#!/usr/bin/env python3
"""AST / line-based translator: CogAlpha pandas factor code -> factor_engine DSL."""
from __future__ import annotations

import re
from dataclasses import dataclass


HARD_MARKERS = ("groupby", "cumcount", "maximum.accumulate", "pd.concat", "for ", "while ")

# talib.*(high, low, close, n) style calls handled via regex pre-pass
TALIB_REWRITES = [
    (
        r"talib\.ATR\(\s*high\.values[^,]*,\s*low\.values[^,]*,\s*close\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)",
        r"ATR(high, low, close, \1)",
    ),
    (
        r"talib\.ATR\(\s*high\.values[^,]*,\s*low\.values[^,]*,\s*close\.values[^,]*,\s*(\d+)\s*\)",
        r"ATR(high, low, close, \1)",
    ),
    (r"talib\.ATR\(\s*high,\s*low,\s*close,\s*timeperiod\s*=\s*(\d+)\s*\)", r"ATR(high, low, close, \1)"),
    (r"talib\.ATR\(\s*high,\s*low,\s*close,\s*(\d+)\s*\)", r"ATR(high, low, close, \1)"),
    (r"talib\.EMA\(\s*([^,]+)\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)", r"EMA(\1, \2)"),
    (r"talib\.EMA\(\s*([^,]+),\s*timeperiod\s*=\s*(\d+)\s*\)", r"EMA(\1, \2)"),
    (r"talib\.SMA\(\s*([^,]+)\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)", r"SMA(\1, \2)"),
    (r"talib\.SMA\(\s*([^,]+),\s*timeperiod\s*=\s*(\d+)\s*\)", r"SMA(\1, \2)"),
    (r"talib\.ROC\(\s*([^,]+)\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)", r"ROC(\1, \2)"),
    (r"talib\.ROC\(\s*([^,]+),\s*timeperiod\s*=\s*(\d+)\s*\)", r"ROC(\1, \2)"),
    (r"talib\.RSI\(\s*([^,]+)\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)", r"RSI(\1, \2)"),
    (r"talib\.RSI\(\s*([^,]+),\s*timeperiod\s*=\s*(\d+)\s*\)", r"RSI(\1, \2)"),
    (r"talib\.ADX\(\s*high\.values[^,]*,\s*low\.values[^,]*,\s*close\.values[^,]*,\s*timeperiod\s*=\s*(\d+)\s*\)", r"ADX(high, low, close, \1)"),
    (r"talib\.ADX\(\s*high,\s*low,\s*close,\s*timeperiod\s*=\s*(\d+)\s*\)", r"ADX(high, low, close, \1)"),
    (r"talib\.ADX\(\s*high,\s*low,\s*close,\s*(\d+)\s*\)", r"ADX(high, low, close, \1)"),
    (r"talib\.TRANGE\(\s*high\.values[^,]*,\s*low\.values[^,]*,\s*close\.values[^,]*\s*\)", r"max(high - low, abs(high - delay(close, 1)), abs(low - delay(close, 1)))"),
    (r"talib\.TRANGE\(\s*high,\s*low,\s*close\s*\)", r"max(high - low, abs(high - delay(close, 1)), abs(low - delay(close, 1)))"),
]

ALPHA_TOOL_PATTERNS = [
    (
        r"(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=\s*alpha_tools\.classify_volume_regime\(\s*([^,]+),\s*window\s*=\s*(\d+)[^)]*\)",
        lambda m: (
            f"{m.group(3)} = protected_div({m.group(4)}, ewm_mean({m.group(4)}, {m.group(5)}))\n"
            f"{m.group(1)} = where(protected_div({m.group(4)}, ewm_mean({m.group(4)}, {m.group(5)})) > 1.5, 1, 0)\n"
            f"{m.group(2)} = where(protected_div({m.group(4)}, ewm_mean({m.group(4)}, {m.group(5)})) < 0.6, 1, 0)"
        ),
    ),
    (
        r"(\w+)\s*=\s*alpha_tools\.classify_volume_regime\(\s*([^,]+),\s*window\s*=\s*(\d+)[^)]*\)",
        lambda m: f"{m.group(1)} = protected_div({m.group(2)}, ewm_mean({m.group(2)}, {m.group(3)}))",
    ),
    (
        r"(\w+)\s*,\s*(\w+)\s*=\s*alpha_tools\.decompose_overnight_intraday\(\s*([^,]+),\s*([^)]+)\)",
        lambda m: (
            f"{m.group(1)} = protected_div({m.group(4)}, delay({m.group(3)}, 1)) - 1\n"
            f"{m.group(2)} = protected_div({m.group(3)}, {m.group(4)}) - 1"
        ),
    ),
]

COLUMN_MAP = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "open_": "open",
    "open_price": "open",
}

SKIP_ASSIGNMENT_MARKERS = (
    ".quantile(",
    "factor.name",
    "lower =",
    "upper =",
)

RESIDUAL_MARKERS = ("df", "pd.", "np.", "talib.", "groupby", "lambda", "axis=", ".index")


@dataclass
class TranslateResult:
    dsl: str
    status: str
    source: str
    notes: str = ""


def _find_matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_matching_open_paren(s: str, close_idx: int) -> int:
    depth = 0
    for i in range(close_idx, -1, -1):
        ch = s[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_call_args(s: str, start: int) -> tuple[list[str], int] | None:
    """Return comma-split args for call starting at `start` (position of name)."""
    open_idx = s.find("(", start)
    if open_idx < 0:
        return None
    close_idx = _find_matching_paren(s, open_idx)
    if close_idx < 0:
        return None
    inner = s[open_idx + 1 : close_idx]
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in inner:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    args.append("".join(cur).strip())
    return args, close_idx + 1


def _rewrite_calls(expr: str, name: str, builder) -> str:
    out = expr
    needle = f"{name}("
    while True:
        idx = out.find(needle)
        if idx < 0:
            break
        parsed = _extract_call_args(out, idx)
        if not parsed:
            break
        args, end = parsed
        replacement = builder(args)
        out = out[:idx] + replacement + out[end:]
    return out


def _rewrite_np_where(expr: str) -> str:
    def build(args: list[str]) -> str:
        if len(args) < 3:
            return "where(0, 0, 0)"
        return f"where({args[0]}, {args[1]}, {args[2]})"

    return _rewrite_calls(expr, "np.where", build)


def _rewrite_series_where(expr: str) -> str:
    out = expr
    guard = 0
    while ".where(" in out and guard < 32:
        guard += 1
        idx = out.find(".where(")
        # walk back to find receiver expression start
        j = idx - 1
        while j >= 0 and out[j].isspace():
            j -= 1
        if j < 0:
            break
        if out[j] == ")":
            close_idx = j
            open_idx = _find_matching_open_paren(out, close_idx)
            if open_idx < 0:
                break
            recv_start = open_idx
            receiver = out[recv_start : close_idx + 1]
        elif out[j].isalnum() or out[j] == "_":
            end = j
            while end >= 0 and (out[end].isalnum() or out[end] == "_"):
                end -= 1
            recv_start = end + 1
            receiver = out[recv_start : j + 1]
        else:
            break
        parsed = _extract_call_args(out, idx + 1)  # position at 'where'
        if not parsed:
            break
        args, call_end = parsed
        if len(args) < 2:
            break
        other = args[1] if len(args) == 2 else args[1]
        repl = f"where({args[0]}, {receiver}, {other})"
        out = out[:recv_start] + repl + out[call_end:]
    return out


def _balanced_expr_before_dot_method(s: str, dot_idx: int) -> tuple[int, int] | None:
    """Return [start, dot_idx) span for receiver of `.method` at dot_idx."""
    j = dot_idx - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    if j < 0:
        return None
    if s[j] == ")":
        close_idx = j
        open_idx = _find_matching_open_paren(s, close_idx)
        if open_idx < 0:
            return None
        recv_start = open_idx
        k = recv_start - 1
        while k >= 0 and s[k].isspace():
            k -= 1
        if k >= 0 and (s[k].isalnum() or s[k] == "_"):
            end = k
            while end >= 0 and (s[end].isalnum() or s[end] == "_"):
                end -= 1
            recv_start = end + 1
        return recv_start, dot_idx
    if s[j].isalnum() or s[j] == "_":
        end = j
        while end >= 0 and (s[end].isalnum() or s[end] == "_"):
            end -= 1
        return end + 1, dot_idx
    return None


def _replace_trailing_method(expr: str, method: str, inner_pat: str, fmt: str, *, suffix: str = "") -> str:
    """Replace `receiver.method(inner)suffix` from right to left with balanced receiver."""
    out = expr
    needle = f".{method}("
    guard = 0
    while needle in out and guard < 64:
        guard += 1
        idx = out.rfind(needle)
        span = _balanced_expr_before_dot_method(out, idx)
        if not span:
            break
        recv_start, dot_idx = span
        receiver = out[recv_start:dot_idx]
        parsed = _extract_call_args(out, dot_idx + 1)
        if not parsed:
            break
        args, end = parsed
        inner = args[0] if args else ""
        m = re.match(inner_pat, inner.strip())
        if not m:
            break
        end2 = end
        if suffix and out[end : end + len(suffix)] == suffix:
            end2 = end + len(suffix)
        repl = fmt.format(receiver=receiver, **{f"g{i}": g for i, g in enumerate(m.groups())})
        out = out[:recv_start] + repl + out[end2:]
    return out


def _replace_rolling_agg(expr: str, agg_suffix: str, op: str) -> str:
    if agg_suffix == ".std()":
        pat = r"\.rolling\(\s*(?:window\s*=\s*)?(\w+)[^)]*\)\.std\([^)]*\)"
    elif agg_suffix == ".rank(pct=True)":
        pat = r"\.rolling\(\s*(?:window\s*=\s*)?(\w+)[^)]*\)\.rank\(\s*pct\s*=\s*True\s*\)"
    else:
        pat = rf"\.rolling\(\s*(?:window\s*=\s*)?(\w+)[^)]*\){re.escape(agg_suffix)}"
    out = expr
    guard = 0
    while re.search(pat, out) and guard < 64:
        guard += 1
        m = list(re.finditer(pat, out))[-1]
        span = _balanced_expr_before_dot_method(out, m.start())
        if not span:
            break
        receiver = out[span[0] : span[1]]
        repl = f"{op}({receiver}, {m.group(1)})"
        out = out[: span[0]] + repl + out[m.end() :]
    return out


def _apply_ts_passes(work: str) -> str:
    work = _strip_noise(work)
    rolling_ops = [
        (".sum()", "ts_sum"),
        (".mean()", "ts_mean"),
        (".std()", "ts_std"),
        (".median()", "ts_median"),
        (".min()", "ts_min"),
        (".max()", "ts_max"),
        (".skew()", "ts_skew"),
        (".rank(pct=True)", "ts_rank"),
    ]
    for _ in range(24):
        prev = work
        for suffix, op in rolling_ops:
            work = _replace_rolling_agg(work, suffix, op)
        work = _replace_trailing_method(work, "ewm", r"^span\s*=\s*(\d+)", "ewm_mean({receiver}, {g0})", suffix=".mean()")
        work = _replace_trailing_method(work, "shift", r"^(\d+)$", "delay({receiver}, {g0})")
        work = _replace_trailing_method(work, "pct_change", r"^(\d+)$", "ts_pct({receiver}, {g0})")
        work = _replace_trailing_method(work, "pct_change", r"^$", "ts_pct({receiver}, 1)")
        work = _replace_trailing_method(work, "diff", r"^$", "ts_delta({receiver}, 1)", suffix="")
        work = _replace_trailing_method(work, "abs", r"^$", "abs({receiver})", suffix="")
        work = re.sub(r"(.+?)\.expanding\(\)\.max\(\)", r"ts_max(\1, 252)", work)
        work = re.sub(r"(.+?)\.rolling\(\s*(\d+)[^)]*\)\.rank\(\s*pct\s*=\s*True\s*\)", r"ts_rank(\1, \2)", work)
        work = re.sub(r"(.+?)\.rolling\(\s*(\d+)[^)]*\)\.corr\(\s*([^)]+)\s*\)", r"ts_corr(\1, \3, \2)", work)
        if work == prev:
            break
    return work


def _normalize_code(code: str) -> str:
    code = code.replace("df_copy['", "df['").replace('df_copy["', 'df["')
    code = code.replace("df_copy.", "df.")
    code = re.sub(r"\(df\.copy\(\)\)", "df", code)
    code = re.sub(
        r"(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=\s*high\s*,\s*low\s*,\s*volume",
        r"\1 = high\n\2 = low\n\3 = volume",
        code,
    )
    code = re.sub(
        r"df\[\[['\"]open['\"],\s*['\"]close['\"]\]\]\.min\(axis\s*=\s*1\)",
        "min(open, close)",
        code,
    )
    code = re.sub(
        r"df\[\[['\"]open['\"],\s*['\"]close['\"]\]\]\.max\(axis\s*=\s*1\)",
        "max(open, close)",
        code,
    )
    code = re.sub(r"pd\.Series\(np\.nan,\s*index=[^)]+\)", "nan", code)
    code = re.sub(r"pd\.Series\(\s*(\w+)\s*,\s*index=[^)]+\)", r"\1", code)
    code = re.sub(r"\bnp\.nan\b", "nan", code)
    code = re.sub(r"(\w+)\*\*2\b", r"pow(\1, 2)", code)
    code = re.sub(r"(\))\*\*2\b", r"pow(\1, 2)", code)
    code = re.sub(r"(\w+)\[(\w+)\]\s*=\s*(\w+)\[\2\]", r"\1 = where(\2, \3, 0)", code)
    code = re.sub(r"~(\w+)", r"not_(\1)", code)
    code = re.sub(r"\.fillna\(\s*0\s*\)", "", code)
    code = re.sub(r"\.replace\(\s*0\.0\s*,\s*np\.nan\s*\)", "", code)
    code = re.sub(r"\.replace\(\s*0\s*,\s*np\.nan\s*\)", "", code)
    code = re.sub(r"\.values\b", "", code)
    code = re.sub(r"(\w+)\.pct_change\(\s*\)\.abs\(\s*\)", r"abs(ts_pct(\1, 1))", code)
    code = re.sub(r"\.astype\(np\.float64\)", "", code)
    code = re.sub(r"(\w+)\.abs\(\s*\)", r"abs(\1)", code)
    code = re.sub(
        r"\blower\s*=\s*min\(open,\s*close\)\s*-\s*low",
        "lower_shadow = min(open, close) - low",
        code,
    )
    code = re.sub(
        r"\bupper\s*=\s*high\s*-\s*max\(open,\s*close\)",
        "upper_shadow = high - max(open, close)",
        code,
    )
    code = re.sub(r"\.rolling\(window=window([^)]*)\)", lambda m: f".rolling(20{m.group(1)})", code)
    code = re.sub(r"\.rolling\(window=(\d+)([^)]*)\)", r".rolling(\1\2)", code)
    code = re.sub(r"\.replace\(\s*\[np\.inf,\s*-np\.inf\]\s*,\s*(?:np\.)?nan\s*\)", "", code)
    code = re.sub(r"\.astype\(\s*float\s*\)", "", code)
    code = re.sub(r"\.astype\(\s*int\s*\)", "", code)
    code = re.sub(r"\.values\.astype\([^)]+\)", "", code)
    code = re.sub(r"df\[['\"](\w+)['\"]\]", lambda m: COLUMN_MAP.get(m.group(1), m.group(1)), code)
    for pattern, repl in TALIB_REWRITES:
        code = re.sub(pattern, repl, code, flags=re.IGNORECASE)
    for pattern, builder in ALPHA_TOOL_PATTERNS:
        code = re.sub(pattern, builder, code)
    guard = 0
    while "pd.Series(" in code and guard < 32:
        guard += 1
        new_code = re.sub(r"pd\.Series\(\s*([^,]+),\s*index=[^)]+\)", r"\1", code, count=1)
        if new_code == code:
            break
        code = new_code
    code = re.sub(r"df\[['\"]style_gate_[^'\"]+['\"]\]\s*>\s*0", "1", code)
    code = re.sub(r"df\[['\"]style_gate_[^'\"]+['\"]\]", "1", code)
    code = re.sub(r"\bstyle_gate_\w+\s*>\s*0", "1", code)
    code = re.sub(r"\bstyle_gate_\w+\b", "1", code)
    return code


def _sub_env(expr: str, env: dict[str, str]) -> str:
    work = expr
    for name in sorted(env, key=len, reverse=True):
        work = re.sub(rf"\b{re.escape(name)}\b", f"({env[name]})", work)
    return work


def _strip_noise(expr: str) -> str:
    work = expr
    work = re.sub(r",\s*min_periods\s*=\s*\d+", "", work)
    work = re.sub(r",\s*adjust\s*=\s*False", "", work)
    work = re.sub(r",\s*ddof\s*=\s*\d+", "", work)
    work = re.sub(r"\.replace\(\s*\[np\.inf,\s*-np\.inf\]\s*,\s*(?:np\.)?nan\s*\)", "", work)
    work = re.sub(r"\.replace\(\s*0\.0\s*,\s*(?:np\.)?nan\s*\)", "", work)
    work = re.sub(r"\.replace\(\s*0\s*,\s*(?:np\.)?nan\s*\)", "", work)
    work = re.sub(r"\.fillna\([^)]*\)", "", work)
    work = re.sub(r"\.clip\([^)]*\)", "", work)
    work = re.sub(r"\.astype\([^)]*\)", "", work)
    work = re.sub(r"\.name\s*=\s*['\"][^'\"]+['\"]", "", work)
    return work


def _translate_expr(expr: str, env: dict[str, str]) -> str:
    work = _sub_env(expr.strip(), env)
    work = re.sub(r"(\w+)\s*\*\*\s*(\d+)", r"pow(\1, \2)", work)
    work = re.sub(r"(\))\s*\*\*\s*(\d+)", r"pow(\1, \2)", work)
    work = _rewrite_np_where(work)
    work = re.sub(r"np\.isnan\(([^)]+)\)", r"is_nan(\1)", work)
    work = re.sub(r"np\.tanh\(([^)]+)\)", r"tanh(\1)", work)
    work = re.sub(r"np\.log\(([^)]+)\)", r"log(\1)", work)
    work = re.sub(r"np\.sqrt\(([^)]+)\)", r"sqrt(\1)", work)
    work = re.sub(r"np\.abs\(([^)]+)\)", r"abs(\1)", work)
    work = re.sub(r"np\.sign\(([^)]+)\)", r"sign(\1)", work)
    work = re.sub(r"np\.maximum\(([^,]+),\s*([^)]+)\)", r"max(\1, \2)", work)
    work = re.sub(r"np\.minimum\(([^,]+),\s*([^)]+)\)", r"min(\1, \2)", work)
    work = re.sub(r"np\.clip\(([^,]+),\s*([^,]+),\s*([^)]+)\)", r"cap(\1, \2, \3)", work)
    work = re.sub(r"np\.nan_to_num\(([^)]+)\)", r"nan_to_num(\1)", work)
    work = _rewrite_series_where(work)

    work = re.sub(r"(\w+)\s*/\s*\((\w+)\s*\+\s*1e-8\)", r"protected_div(\1, \2)", work)
    work = re.sub(r"(\w+)\s*/\s*\((\w+)\s*\+\s*1\.0e-8\)", r"protected_div(\1, \2)", work)
    work = re.sub(r"protected_div\(([^,]+),\s*0\)", r"protected_div(\1, 1)", work)
    work = work.replace("++", "+").replace("+-", "-")
    work = _apply_ts_passes(work)
    work = _replace_trailing_method(work, "rank", r"^$", "ts_rank({receiver}, 20)")
    work = _replace_trailing_method(work, "rank", r"^pct\s*=\s*True$", "ts_rank({receiver}, 20)")
    if any(tok in work for tok in (".rolling(", ".ewm(", ".shift(", ".pct_change(", ".rank(")):
        work = _apply_ts_passes(work)
        work = _replace_trailing_method(work, "rank", r"^$", "ts_rank({receiver}, 20)")
    return work.strip()


def _extract_return_expr(code: str) -> str | None:
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("return "):
            return line[len("return ") :].strip()
    return None


def _should_skip_assignment(line: str) -> bool:
    if any(m in line for m in SKIP_ASSIGNMENT_MARKERS):
        return True
    if re.search(r"\bfactor\s*=\s*factor\.clip\(", line):
        return True
    return False


def translate_python(code: str, tools: str = "") -> TranslateResult:
    raw = code
    if any(m in raw for m in HARD_MARKERS):
        return TranslateResult("", "hard", "ast", "uses groupby/cumcount/concat/cummax")

    work = _normalize_code(raw)
    env: dict[str, str] = {}

    for line in work.splitlines():
        line = line.strip()
        if not line or line.startswith(("def ", "#", '"""', "'''")):
            continue
        if line in ("df = df.copy()",) or line.startswith("df.copy("):
            continue
        if _should_skip_assignment(line):
            continue
        if re.fullmatch(r"window\s*=\s*\d+", line):
            lhs, rhs = line.split("=", 1)
            env[lhs.strip()] = rhs.strip()
            continue
        if "=" not in line or line.startswith("return "):
            continue
        lhs, rhs = line.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs.startswith("df['") or lhs.startswith('df["'):
            var = lhs.split("'")[1] if "'" in lhs else lhs.split('"')[1]
        else:
            var = lhs
        if var in COLUMN_MAP and rhs in COLUMN_MAP.values():
            env[var] = rhs
            continue
        if rhs in ("close", "open", "high", "low", "volume"):
            env[var] = rhs
            continue
        try:
            env[var] = _translate_expr(rhs, env)
        except Exception:
            return TranslateResult("", "needs_review", "ast", f"failed assignment: {var}")

    ret = _extract_return_expr(work)
    if ret is None:
        return TranslateResult("", "needs_review", "ast", "no return statement")

    ret = re.sub(r"df\[['\"][^'\"]+['\"]\]", "", ret)
    ret = ret.strip()
    if ret in env:
        dsl = env[ret]
    else:
        dsl = _translate_expr(ret, env)

    if not dsl or any(tok in dsl for tok in RESIDUAL_MARKERS):
        return TranslateResult("", "needs_review", "ast", "residual python in expression")

    return TranslateResult(dsl, "ready", "ast")


def dsl_to_lqtp(dsl: str) -> str:
    """Best-effort factor_engine DSL -> LQTP formula."""
    out = dsl
    out = re.sub(r"ts_pct\(([^,]+),\s*(\d+)\)", r"(\1 / delay(\1, \2) - 1)", out)
    out = re.sub(r"protected_div\(", "safe_div(", out)
    out = re.sub(r"ewm_mean\(", "ema(", out)
    out = re.sub(r"ts_median\(([^,]+),\s*(\d+)\)", r"ts_quantile(\1, \2, 0.5)", out)
    out = re.sub(r"add\(([^,]+),\s*1\)", r"(\1 + 1)", out)
    return out
