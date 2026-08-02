#!/usr/bin/env python3
"""Materialize CogAlpha DSL factors via factor_engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.dsl_parser import parse_factor  # noqa: E402
from api.factor import Factor  # noqa: E402
from api.mining_integration import validate_factor_engine_dsl  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from storage.cache import CacheManager  # noqa: E402
from storage.factor_format import series_to_long_table  # noqa: E402

from scripts.cogalpha_lqtp.ast_translator import (  # noqa: E402
    _replace_func_calls,
    dsl_to_lqtp,
    lqtp_to_fe_dsl,
)
from scripts.cogalpha_lqtp.data_access_panel import (  # noqa: E402
    ashare_materialize_data_source_config,
    build_materialize_backend,
    build_materialize_data_source,
)

# Broken / multi-statement pack formulas → single LQTP-friendly expression.
EXTERNAL_FORMULA_FIXES: dict[str, str] = {
    "factor_directional_vol_adaptive_crossover": (
        "ema((high - low) / close, 5) * "
        "where(volume / ts_mean(volume, 20) > 0.9, "
        "ema(volume / ts_mean(volume, 20), 10), "
        "ema(volume / ts_mean(volume, 20), 30)) * "
        "((close - open) / close)"
    ),
    "factor_volume_adjusted_momentum": (
        "ema(ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)), 5)"
    ),
    "factor_reversal_intraday_5d_ema": "-ema((close - open) / open, 5)",
}


def _ashare_smoke_data_source(
    *,
    parquet_root: Path,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    cfg = ashare_materialize_data_source_config(
        start_date=start_date,
        end_date=end_date,
        max_files=10,
        parquet_root=parquet_root,
    )
    return cfg


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


def _scan_balanced(text: str, start: int) -> int:
    """Return index after balanced parens starting at ``text[start] == '('``."""
    if start >= len(text) or text[start] != "(":
        return start
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _rewrite_scalar_minmax(text: str) -> str:
    """Rewrite ``min/max(literal, expr)`` to nested ``cap`` (compat-safe at runtime)."""
    import re

    fn_pat = re.compile(r"\b(min|max)\s*\(")

    def _rewrite_once(src: str) -> tuple[str, bool]:
        m = fn_pat.search(src)
        if not m:
            return src, False
        fn = m.group(1)
        open_idx = m.end() - 1
        close_idx = _scan_balanced(src, open_idx)
        if close_idx <= open_idx:
            return src, False
        inner = src[open_idx + 1 : close_idx - 1]
        args = _split_top_level_args(inner)
        if len(args) != 2 or not re.fullmatch(r"[\d.]+", args[0]):
            inner_new, changed = _rewrite_once(inner)
            if not changed:
                return src, False
            repl = f"{fn}({inner_new})"
            return src[: m.start()] + repl + src[close_idx:], True
        scalar, expr = args[0], args[1]
        inner_expr, _ = _rewrite_once(expr)
        if fn == "max":
            repl = f"cap({inner_expr}, {scalar}, 1e18)"
        else:
            repl = f"cap({inner_expr}, -1e18, {scalar})"
        return src[: m.start()] + repl + src[close_idx:], True

    out = text
    for _ in range(64):
        out, changed = _rewrite_once(out)
        if not changed:
            break
    return out


def _rewrite_sigmoid(text: str) -> str:
    import re

    pat = re.compile(r"\bsigmoid\s*\(")
    i = 0
    out: list[str] = []
    while i < len(text):
        m = pat.match(text, i)
        if m:
            open_idx = m.end() - 1
            close_idx = _scan_balanced(text, open_idx)
            inner = text[open_idx + 1 : close_idx - 1]
            out.append(f"(1 / (1 + exp(-({inner}))))")
            i = close_idx
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _rewrite_safe_div_named_args(text: str) -> str:
    """``safe_div(numerator=A, denominator=B)`` → ``safe_div(A, B)`` (LQTP requires positional)."""
    import re

    def _replacer(args: list[str]) -> str | None:
        num = den = None
        positional: list[str] = []
        for a in args:
            s = a.strip()
            if re.match(r"^numerator\s*=", s):
                num = s.split("=", 1)[1].strip()
            elif re.match(r"^denominator\s*=", s):
                den = s.split("=", 1)[1].strip()
            else:
                positional.append(s)
        if num is not None and den is not None:
            return f"safe_div({num}, {den})"
        if len(positional) == 2 and num is None and den is None:
            return None
        return None

    prev = None
    out = text
    while out != prev:
        prev = out
        out = _replace_func_calls(out, "safe_div", _replacer)
    return out


def _rewrite_span_kwargs(text: str) -> str:
    """``ema(x, span=5)`` / ``ewm(x, span=5)`` → ``ema(x, 5)``."""
    import re

    def _ema_replacer(args: list[str]) -> str | None:
        if len(args) != 2:
            return None
        span_m = re.match(r"^span\s*=\s*(.+)$", args[1].strip())
        if not span_m:
            return None
        return f"ema({args[0]}, {span_m.group(1).strip()})"

    out = text.replace("ewm(", "ema(")
    prev = None
    while out != prev:
        prev = out
        out = _replace_func_calls(out, "ema", _ema_replacer)
    return out


def _expand_assignment_dsl(text: str) -> str:
    """Expand ``a = ...; b = f(a); factor = ...`` into a single expression."""
    import re

    if ";" not in text:
        return text
    parts = [p.strip() for p in text.split(";") if p.strip()]
    env: dict[str, str] = {}
    last_var = ""
    for part in parts:
        # ``if cond then x else y`` is not an assignment
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", part)
        if not m or re.search(r"\bthen\b", part):
            continue
        var, expr = m.group(1), m.group(2).strip()
        for key, val in env.items():
            expr = re.sub(rf"\b{re.escape(key)}\b", f"({val})", expr)
        # if-then-else → where
        ite = re.match(
            r"^if\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)$",
            expr,
            flags=re.I | re.S,
        )
        if ite:
            expr = f"where({ite.group(1).strip()}, {ite.group(2).strip()}, {ite.group(3).strip()})"
        env[var] = expr
        last_var = var
    if "factor" in env:
        return env["factor"]
    if last_var and last_var in env:
        return env[last_var]
    return text


def _normalize_dsl_for_fe(dsl: str, *, factor_name: str = "") -> str:
    """Align mixed LQTP/FE naming before validation (EMA→ema, clip↔cap handled elsewhere)."""
    import re

    text = (dsl or "").strip()
    if not text:
        return text
    if factor_name and factor_name in EXTERNAL_FORMULA_FIXES:
        text = EXTERNAL_FORMULA_FIXES[factor_name]
    text = _expand_assignment_dsl(text)
    text = _rewrite_safe_div_named_args(text)
    text = _rewrite_span_kwargs(text)
    text = re.sub(r"\bEMA\s*\(", "ema(", text)
    text = re.sub(r"\bSMA\s*\(", "ts_mean(", text)
    text = re.sub(r"\bshift\s*\(", "delay(", text)
    text = re.sub(r"\bts_atr\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)", r"ts_mean(true_range(\1, \2, \3), \4)", text)
    text = re.sub(r"\bmedian\s*\(\s*([^,]+)\s*,\s*([^)]+)\)", r"ts_quantile(\1, \2, 0.5)", text)
    # Bare true_range inside ts_mean/ts_* → true_range(high, low, close)
    text = re.sub(
        r"\b(ts_mean|ts_std|ts_sum|ts_max|ts_min|ts_rank|ts_quantile)\s*\(\s*true_range\s*,",
        r"\1(true_range(high, low, close),",
        text,
    )
    text = _rewrite_scalar_minmax(text)
    text = _rewrite_sigmoid(text)
    # Derived field aliases (not physical columns on ashare / LQTP)
    text = re.sub(r"\bintraday_return\b", "((close - open) / open)", text)
    text = re.sub(r"\bintraday_ret\b", "((close - open) / open)", text)
    text = re.sub(r"\bdaily_return\b", "ts_pct(close, 1)", text)
    # LQTP/Alphasage field aliases → factor_engine ashare PV names
    for old, new in (
        ("pre_close", "preclose"),
        ("high_limit", "high_limit"),
        ("low_limit", "low_limit"),
    ):
        text = re.sub(rf"\b{old}\b", new, text)
    return text


def _pick_fe_dsl_and_surface(dsl: str, *, factor_name: str = "") -> tuple[str, str]:
    """Return (dsl_text, surface) for factor_engine parse/run."""
    raw = _normalize_dsl_for_fe(dsl, factor_name=factor_name)
    fe = _normalize_dsl_for_fe(lqtp_to_fe_dsl(raw), factor_name=factor_name)
    lqtp = _normalize_dsl_for_fe(dsl_to_lqtp(fe), factor_name=factor_name)
    for candidate, surface in ((raw, "compat"), (lqtp, "compat"), (fe, "compat"), (fe, "daily")):
        ok, _ = validate_factor_engine_dsl(candidate, surface=surface)
        if ok:
            return candidate, surface
    return raw or fe, "compat"


def materialize_factor(
    *,
    factor_id: str,
    dsl: str,
    data_source_cfg: dict[str, Any],
    lake_root: Path,
    allowed_symbols: set[str] | None = None,
) -> Path:
    fe_dsl, surface = _pick_fe_dsl_and_surface(dsl, factor_name=factor_id)
    ok, msg = validate_factor_engine_dsl(fe_dsl, surface=surface)
    if not ok:
        raise RuntimeError(f"DSL invalid for {factor_id}: {msg}")

    data_source = build_materialize_data_source(data_source_cfg)
    engine = FactorEngine(
        backend=build_materialize_backend(),
        data_source=data_source,
        cache=CacheManager(),
    )
    factor = parse_factor(
        fe_dsl,
        name=factor_id,
        freq="1d",
        universe="A_SHARE_ALL_A_EX_ST",
        surface=surface,
    )
    out = engine.run(factor)
    series = out["result"]
    long_df = series_to_long_table(series)
    long_df["asset"] = long_df["asset"].map(_normalize_symbol)
    long_df = long_df.dropna(subset=["value"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df[np.isfinite(long_df["value"].to_numpy(dtype="float64", copy=False))]
    if allowed_symbols:
        long_df = long_df[long_df["asset"].isin(allowed_symbols)]
        if long_df.empty:
            raise RuntimeError(f"{factor_id}: no rows after LQTP universe filter")
    if long_df.empty:
        raise RuntimeError(f"{factor_id}: no finite factor values after sanitize")

    out_dir = lake_root / factor_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "values.parquet"
    long_df.to_parquet(out_path, index=False)
    return out_path


def load_catalog(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize CogAlpha DSL factors")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lake-root", type=Path, required=True)
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--only", nargs="*", default=None, help="factor_id or function_name subset")
    parser.add_argument("--status", default="ready")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    if args.only:
        only = set(args.only)
        catalog = [
            x
            for x in catalog
            if x["factor_id"] in only or x["function_name"] in only
        ]
    else:
        catalog = [x for x in catalog if x.get("status") == args.status and x.get("dsl")]

    data_cfg = _ashare_smoke_data_source(
        parquet_root=args.data_root,
        start_date=args.start,
        end_date=args.end,
    )
    args.lake_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for entry in catalog:
        factor_id = entry["function_name"]
        dsl = entry["dsl"]
        print(f"materialize {factor_id}: {dsl}")
        out_path = materialize_factor(
            factor_id=factor_id,
            dsl=dsl,
            data_source_cfg=data_cfg,
            lake_root=args.lake_root,
        )
        manifest.append(
            {
                "factor_id": entry["factor_id"],
                "function_name": factor_id,
                "dsl": dsl,
                "values_path": str(out_path),
                "rows": int(pd.read_parquet(out_path).shape[0]),
            }
        )

    manifest_path = args.lake_root / "materialize_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done {len(manifest)} factors -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
