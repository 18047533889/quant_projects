#!/usr/bin/env python3
"""Post-process auto-translated DSL to fix common merge artifacts."""
from __future__ import annotations

import re


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
    return out.strip()
