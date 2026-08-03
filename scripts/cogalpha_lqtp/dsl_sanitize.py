#!/usr/bin/env python3
"""Post-process auto-translated DSL to fix common merge artifacts."""
from __future__ import annotations

import re


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


def simplify_redundant_parens(dsl: str) -> str:
    """Remove layers of ((expr)) that do not change meaning."""
    out = dsl
    changed = True
    guard = 0
    while changed and guard < 64:
        changed = False
        guard += 1
        i = 0
        buf: list[str] = []
        while i < len(out):
            if out[i] == "(" and i + 1 < len(out) and out[i + 1] == "(":
                inner_close = _find_matching_paren(out, i + 1)
                outer_close = _find_matching_paren(out, i)
                if (
                    inner_close > 0
                    and outer_close == inner_close + 1
                    and out[inner_close] == ")"
                ):
                    # ((X)) → (X)
                    buf.append(out[i + 1 : outer_close])
                    i = outer_close + 1
                    changed = True
                    continue
            buf.append(out[i])
            i += 1
        out = "".join(buf)
    # Strip useless parens around bare identifiers / numbers: (close) → close
    out = re.sub(r"\(\s*([A-Za-z_][\w]*)\s*\)", r"\1", out)
    out = re.sub(r"\(\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\)", r"\1", out, flags=re.I)
    return out


def sanitize_dsl(dsl: str) -> str:
    out = dsl
    out = re.sub(r"\bnan\b", "0", out)
    out = re.sub(r"where\(\s*1\s*>\s*0\s*,\s*([^,]+),\s*([^)]+)\)", r"\1", out)
    out = re.sub(r"where\(\(\s*1\s*>\s*0\s*\)\s*,\s*([^,]+),\s*([^)]+)\)", r"\1", out)
    out = re.sub(r"where\(\(\s*1\s*\)\s*,\s*([^,]+),\s*([^)]+)\)", r"\1", out)
    out = re.sub(r"ts_pctabs\(([^,]+),\s*(\d+)\)", r"abs(ts_pct(\1, \2))", out)
    for op in ("mean", "std", "max", "min", "sum", "median"):
        out = re.sub(
            rf"ts_{op}delay\((.+),\s*(\d+)\),\s*(\d+)\)",
            rf"delay(ts_{op}(\1, \2), \3)",
            out,
        )
    out = re.sub(r"\.rolling\(\((\d+)\)=\(\1\)\)\.mean\(\)", "", out)
    out = re.sub(r"\.rolling\(\d+\)\.", "", out)
    out = re.sub(r"(\w+)\.rolling\(\d+\)", r"ts_mean(\1, 20)", out)
    out = re.sub(r"\(([^()]+)\)\.rank\(\)", r"ts_rank(\1, 20)", out)
    out = re.sub(r"ts_pct\(([^,]+),\s*(\d+),\s*(\d+)\)", r"ts_pct(\1, \2)", out)
    out = re.sub(r"\bstyle_gate_\w+\s*>\s*0", "1", out)
    out = re.sub(r"\bstyle_gate_\w+\b", "1", out)
    out = re.sub(r"\bvol\b", "volume", out)
    out = simplify_redundant_parens(out)
    return out.strip()
