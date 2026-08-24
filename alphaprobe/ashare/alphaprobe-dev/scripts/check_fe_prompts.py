#!/usr/bin/env python3
"""校验 AlphaPROBE LLM 提示词示例均可被 factor_engine 解析。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT.parents[3] / "factor_engine"  # quant_projects/factor_engine
# workspace: .../alphaprobe/ashare/alphaprobe-dev → parents[3]=quant_projects
if not (FE / "api").is_dir():
    FE = Path.home() / "quant_projects" / "factor_engine"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FE))

from factor_engine.api.dsl_parser import parse_expr  # noqa: E402
from alphaprobe.fe_bridge.expr_parse import (  # noqa: E402
    looks_like_legacy_alphagen,
    parse_mining_expression,
)

EXAMPLES = [
    "divide(subtract(high, low), add(subtract(high, low), 0.001))",
    "rank(ts_mean(close, 20))",
    "rank(safe_div_null(ts_std(close, 20), ts_mean(close, 20)))",
    "subtract(rank(ts_delta(close, 5)), rank(ts_delta(volume, 10)))",
    "if_else(gt(close, ts_mean(close, 20)), rank(ts_delta(close, 1)), rank(ts_delta(close, 5)))",
    # multi-window
    "rank(subtract(ts_mean(close, 5), ts_mean(close, 60)))",
    "rank(safe_div_null(ts_std(close, 10), ts_mean(close, 60)))",
    "subtract(rank(ts_delta(close, 5)), rank(ts_delta(close, 60)))",
    "rank(subtract(ts_rank(close, 20), ts_rank(close, 120)))",
    "if_else(gt(ts_mean(close, 5), ts_mean(close, 20)), rank(ts_delta(volume, 10)), rank(ts_delta(volume, 60)))",
    # expanded operator coverage (V9 / factor_engine compat)
    "and_(gt(close, open), lt(close, high))",
    "or_(gt(close, open), lt(low, high))",
    "ema(close, 20)",
    "ts_ema(close, 20)",
    "inverse(close)",
    "returns(close)",
    "rank(RSI_WILDER(close, 14))",
    "rank(ADX(high, low, close, 14))",
    "rank(ATR_WILDER(high, low, close, 14))",
    "rank(MACD_hist(close, 12, 26, 9))",
    "rank(ts_zscore(ts_log_return(close, 1), 60))",
    "rank(ts_sharpe(ts_pct(close, 1), 60))",
    "rank(ts_max_drawdown(close, 120))",
    "rank(ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 40))",
    "rank(ts_partial_corr(close, volume, ts_mean(volume, 20), 60))",
    "cs_mad_zscore(ts_delta(close, 5))",
    "rank(ts_topk_mean(volume, 20, 5))",
    "rank(ts_count_if(gt(close, ts_delay(close, 1)), 20))",
    "where(gt(ts_mean(close, 5), ts_mean(close, 60)), rank(volume), neg(rank(volume)))",
    "rank(safe_div_null(ATR_WILDER(high, low, close, 14), close))",
    "rank(ts_quantile(ts_pct(close, 1), 60, 0.9))",
    "rank(ts_beta(ts_pct(close, 1), ts_pct(volume, 1), 40))",
    "zscore(ts_decay_linear(ts_pct(close, 1), 20))",
    "tanh(ts_corr(ts_pct(close, 1), ts_log_return(add(volume, 1.0), 1), 60))",
]

LEGACY = ["$close", "TsMean(close, 5)", "Div(close, open)", "Mul(close, volume)", "Ref(close, 1)"]


def main() -> int:
    prompts = (ROOT / "configs" / "prompts").read_text(encoding="utf-8") if False else ""
    # ensure prompt files mention factor_engine and forbid AlphaGen
    pdir = ROOT / "configs" / "prompts"
    for name in (
        "system_head.txt",
        "features_operators_fe_dsl.txt",
        "generation.txt",
        "validity_fe_dsl.txt",
    ):
        text = (pdir / name).read_text(encoding="utf-8")
        if "factor_engine" not in text.lower() and "FactorEngine" not in text:
            # system_head uses FactorEngine wording
            if "FactorEngine" not in text and "factor_engine" not in text:
                print(f"[warn] {name} missing factor_engine mention")
        if "AlphaGen" not in text and name != "compare_fe_dsl.txt":
            print(f"[warn] {name} should explicitly forbid AlphaGen")

    fail = 0
    for e in EXAMPLES:
        try:
            parse_expr(e, surface="compat")
            parse_mining_expression(e)
            print("OK", e)
        except Exception as exc:
            fail += 1
            print("FAIL", e, exc)

    for bad in LEGACY:
        if not looks_like_legacy_alphagen(bad):
            fail += 1
            print("FAIL legacy detector missed", bad)
            continue
        try:
            parse_mining_expression(bad)
            fail += 1
            print("FAIL should reject", bad)
        except ValueError:
            print("REJECT OK", bad)

    # prompt file must not list bare `and(` / `or(` as allowed ops
    allow = (pdir / "features_operators_fe_dsl.txt").read_text(encoding="utf-8")
    if re.search(r"(?m)^\s*and\s*\(", allow) or re.search(r"(?m)^\s*or\s*\(", allow):
        fail += 1
        print("FAIL prompt lists illegal and(/or(")

    must_mention = [
        "RSI_WILDER",
        "ADX",
        "ATR_WILDER",
        "MACD_hist",
        "ts_sharpe",
        "ts_max_drawdown",
        "ts_partial_corr",
        "cs_mad_zscore",
        "ts_topk_mean",
        "ts_count_if",
        "pre_close",
    ]
    for token in must_mention:
        if token not in allow:
            fail += 1
            print("FAIL prompt missing token", token)

    # verify section-5 names are all parseable as operators (exist in compat allowlist)
    from factor_engine.api.mining_integration import build_dsl_allowlist

    compat = build_dsl_allowlist(surface="compat")
    # extract names from "## 5." section
    sec = allow.split("## 5.", 1)
    if len(sec) < 2:
        fail += 1
        print("FAIL missing section 5 operator list")
    else:
        body = sec[1].split("## 6.", 1)[0]
        names = re.findall(
            r"\b(?:[a-z_][a-z0-9_]*|RSI_WILDER|ADX|ATR_WILDER|MACD_line|MACD_signal|MACD_hist)\b",
            body,
        )
        # filter non-ops words
        skip = {"and", "or", "section", "complete", "list", "only", "use", "following", "names"}
        uniq = sorted({n for n in names if n not in skip})
        missing = [n for n in uniq if n not in compat]
        print("section5_ops", len(uniq), "missing_in_compat", missing)
        if missing:
            fail += 1
            print("FAIL section5 ops not in compat:", missing)

    # coverage vs V9 A ops
    import json
    from pathlib import Path as P

    core_path = P.home() / "quant_projects/cold_start_library/library/production_default_core_v9.json"
    if core_path.is_file():
        core = json.loads(core_path.read_text())
        v9 = set()
        for row in core:
            if row.get("market") != "A":
                continue
            for o in str(row.get("operators_latest") or row.get("operators") or "").split(","):
                o = o.strip()
                if o:
                    v9.add(o)
        prompt_ops = set(re.findall(
            r"\b(?:[a-z_][a-z0-9_]*|RSI_WILDER|ADX|ATR_WILDER|MACD_line|MACD_signal|MACD_hist)\b",
            allow,
        ))
        not_in_prompt = sorted(v9 - prompt_ops)
        print("v9_ops", len(v9), "missing_in_prompt", len(not_in_prompt), not_in_prompt[:20])
        if not_in_prompt:
            fail += 1
            print("FAIL V9 ops not mentioned in prompt:", not_in_prompt)

    print("result", "PASS" if fail == 0 else f"FAIL({fail})")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
