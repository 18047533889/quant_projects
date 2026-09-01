#!/usr/bin/env python3
"""将 V9 backend-audited 库转为 AlphaPROBE 可用的 A 股冷启动 yaml，并预计算 IC。

输出：
  data/ashare/backend_v9_core.yaml          # AlphaPROBE 直接读
  data/ashare/backend_v9_metrics.jsonl      # 每条执行/IC 明细
  data/ashare/backend_v9_precompute_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FE = ROOT.parent / "factor_engine"
DA = ROOT.parent / "data_access"
for p in (str(SRC), str(FE), str(DA)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 默认不进 A 股 AlphaPROBE 主库的字段（无估值/资本数据源）
EXCLUDE_FIELDS = {
    "circulating_cap",
    "turnover_ratio",
    "exchange",
    "fiscal_quarter",
    "revenue",
    "market_cap",
    "net_income",
}

# A 股 parquet Return 为 bp；AlphaPROBE/data_access 当前读路径未必归一，导出时改写为小数收益
_DECIMAL_RET = "(safe_div_null(close, pre_close) - 1.0)"
_COALESCE_RET_RE = re.compile(
    r"coalesce\s*\(\s*ret\s*,\s*safe_div_null\s*\(\s*close\s*,\s*pre_close\s*\)\s*-\s*1(?:\.0)?\s*\)"
)
_RET_TOKEN_RE = re.compile(r"\bret\b")
_LOCAL_ASHARE = Path.home() / "quant_projects" / "data" / "a_share" / "lqtp_data"


class _RetDecimalProxy:
    """把 bp 收益列 ret/Return 归一为小数，供 FactorEngine 计算。"""

    def __init__(self, inner: Any, scale: float = 1.0 / 10000.0) -> None:
        self._inner = inner
        self._scale = float(scale)

    def load_column(self, name: str):
        series = self._inner.load_column(name)
        if name in {"ret", "Return"}:
            return series * self._scale
        return series

    def load_columns(self, names: list[str]):
        out = self._inner.load_columns(names)
        for name in names:
            if name in {"ret", "Return"} and name in out:
                out[name] = out[name] * self._scale
        return out

    def __getattr__(self, item: str):
        val = getattr(self._inner, item)
        import datetime as _dt
        if item == "root" and isinstance(val, os.PathLike):
            # engine 的 scope 身份编码（DA-P0-011 strict 模式）不接受 PathLike
            val = str(val)
        elif item in {"start_date", "end_date"} and isinstance(val, _dt.datetime) and val.tzinfo is None:
            # R32-P0-107: strict 模式要求 tz-aware datetime
            val = val.replace(tzinfo=_dt.timezone.utc)
        return val


def rewrite_ret_to_decimal(formula: str) -> str:
    text = _COALESCE_RET_RE.sub(_DECIMAL_RET, formula)
    return _RET_TOKEN_RE.sub(_DECIMAL_RET, text)


def _fields_of(row: dict[str, Any]) -> set[str]:
    raw = row.get("fields_latest") or row.get("fields") or ""
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def _ops_of(row: dict[str, Any]) -> list[str]:
    raw = row.get("operators_latest") or row.get("operators") or ""
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _topic(row: dict[str, Any]) -> str:
    fam = str(row.get("family") or "pv").strip() or "pv"
    return f"V9/{fam}"


def _desc(row: dict[str, Any]) -> str:
    expl = str(row.get("explanation") or "").strip()
    fid = str(row.get("factor_id") or "").strip()
    if expl:
        return f"{fid}: {expl}" if fid else expl
    return fid or "V9 cold-start factor"


def select_ashare_rows(library_json: Path) -> list[dict[str, Any]]:
    rows = json.loads(library_json.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("market") or "").upper() != "A":
            continue
        if row.get("v9_default_pool") not in (None, "core") and row.get("availability_tier") not in (
            None,
            "core",
            "derived",
        ):
            # production_default_core 已是 core；兼容全量 accepted
            pass
        fields = _fields_of(row)
        if fields & EXCLUDE_FIELDS:
            continue
        formula = str(row.get("formula_v9") or row.get("formula_latest") or "").strip()
        if not formula:
            continue
        if row.get("v9_pandas_runtime") not in (None, "pass"):
            continue
        out.append(row)
    return out


def _panel_pearson_ic(f_mat: np.ndarray, y_mat: np.ndarray, *, min_cs: int = 30) -> tuple[float | None, float | None]:
    """矩阵版按日截面 Pearson IC。f_mat/y_mat: [n_days, n_stocks]。"""
    ics: list[float] = []
    n_days = min(f_mat.shape[0], y_mat.shape[0])
    for i in range(n_days):
        a = f_mat[i]
        b = y_mat[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_cs:
            continue
        aa = a[mask]
        bb = b[mask]
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        da = float(np.dot(aa, aa))
        db = float(np.dot(bb, bb))
        if da < 1e-18 or db < 1e-18:
            continue
        c = float(np.dot(aa, bb) / math.sqrt(da * db))
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return None, None
    arr = np.asarray(ics, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    icir = float(mean / std) if std and np.isfinite(std) and std > 1e-12 else None
    return mean, icir


def make_fwd_matrix(ds, label_days: int) -> tuple[pd.Index, pd.Index, np.ndarray]:
    vwap = ds.load_column("vwap")
    fwd = vwap.groupby(level=1).shift(-label_days) / vwap - 1.0
    wide = fwd.unstack(level=1)
    dates = wide.index
    stocks = wide.columns
    return dates, stocks, wide.to_numpy(dtype=float)


def build_engine(start: str, end: str, *, prefer_parquet: bool = True):
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import build_data_source

    os.environ.setdefault("ASHARE_PARQUET_ROOT", str(_LOCAL_ASHARE))
    fields = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "Vwap",
        "amount": "Amount",
        "ret": "Return",
        "pre_close": "PreClose",
        "preclose": "PreClose",
        "factor": "Factor",
    }
    if prefer_parquet:
        cfg: dict[str, Any] = {
            "type": "parquet",
            "root": str(_LOCAL_ASHARE / "StockDailyBar"),
            "timestamp_col": "TradeDate",
            "instrument_col": "Symbol",
            "fields": fields,
            "start_date": start,
            "end_date": end,
        }
    else:
        from api.mining_integration import default_ashare_pv_data_source_config

        cfg = default_ashare_pv_data_source_config(start_date=start, end_date=end)

    raw_ds = build_data_source(cfg)
    ds = _RetDecimalProxy(raw_ds)
    eng = FactorEngine(backend=build_backend("pandas"), data_source=ds, run_mode="research")
    return eng, ds


def precompute_one(
    eng,
    formula: str,
    *,
    dates: pd.Index,
    stocks: pd.Index,
    y_mat: np.ndarray,
    label_days: int,
) -> dict[str, Any]:
    from api.dsl_parser import parse_factor

    t0 = time.time()
    fac = parse_factor(formula, name="cs", surface="compat")
    result = eng.run(fac)["result"]
    if not isinstance(result, pd.Series):
        raise TypeError(type(result))
    s = pd.to_numeric(result, errors="coerce")
    vals = s.to_numpy(dtype=float)
    finite = int(np.isfinite(vals).sum())
    total = int(len(vals))
    finite_ratio = finite / total if total else 0.0
    wide = s.unstack(level=1).reindex(index=dates, columns=stocks)
    f_mat = wide.to_numpy(dtype=float)
    ic, icir = _panel_pearson_ic(f_mat, y_mat)
    uniq = int(np.unique(vals[np.isfinite(vals)]).size) if finite else 0
    return {
        "status": "pass",
        "finite_ratio": finite_ratio,
        "n_finite": finite,
        "n_total": total,
        "n_unique": uniq,
        "ic": ic,
        "icir": icir,
        "abs_ic": abs(ic) if ic is not None else None,
        "label_days": label_days,
        "elapsed_sec": round(time.time() - t0, 4),
    }


def export_yaml(
    entries: list[dict[str, Any]],
    out_path: Path,
    *,
    schema_note: str,
    values_precomputed: bool,
) -> None:
    payload = {
        "schema_version": "cold_start.backend_v9.ashare",
        "quality_policy": "factorengine_backend_audited_v9_precomputed",
        "market": "ashare",
        "frequency": "daily",
        "domain": "price_volume",
        "expression_syntax": "factor_engine_dsl",
        "operator_policy": "lqtp_pv_daily",
        "dsl_validated": True,
        "values_precomputed": bool(values_precomputed),
        "source": schema_note,
        "entry_count": len(entries),
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--library-json",
        type=Path,
        default=ROOT / "library" / "production_default_core_v9.json",
    )
    ap.add_argument("--out-yaml", type=Path, default=ROOT / "data" / "ashare" / "backend_v9_core.yaml")
    ap.add_argument("--metrics-jsonl", type=Path, default=ROOT / "data" / "ashare" / "backend_v9_metrics.jsonl")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "data" / "ashare" / "backend_v9_precompute_summary.json",
    )
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--label-days", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 条")
    ap.add_argument("--min-finite-ratio", type=float, default=0.05)
    ap.add_argument("--skip-precompute", action="store_true", help="只做 parse+导出，不算 IC")
    ap.add_argument("--parse-only-fail-fast", action="store_true")
    args = ap.parse_args()

    from api.dsl_parser import parse_expr
    from cold_start_library.runtime.dsl import extract_dsl_operator_names

    rows = select_ashare_rows(args.library_json)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[select] ashare candidates: {len(rows)} from {args.library_json}")

    parse_ok: list[dict[str, Any]] = []
    parse_fail: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("formula_v9") or "").strip()
        formula = rewrite_ret_to_decimal(raw)
        try:
            parse_expr(formula, surface="compat")
            row = dict(row)
            row["formula_v9"] = formula
            row["formula_v9_raw"] = raw
            parse_ok.append(row)
        except Exception as exc:
            parse_fail.append({"factor_id": row.get("factor_id"), "error": f"{type(exc).__name__}: {exc}"})
            if args.parse_only_fail_fast:
                raise
    print(f"[parse] ok={len(parse_ok)} fail={len(parse_fail)}")

    metrics_by_id: dict[str, dict[str, Any]] = {}
    exec_fail: list[dict[str, Any]] = []

    if not args.skip_precompute:
        print(f"[precompute] parquet ashare {args.start}..{args.end}, label={args.label_days}")
        eng, ds = build_engine(args.start, args.end)
        dates, stocks, y_mat = make_fwd_matrix(ds, args.label_days)
        print(f"[precompute] panel days={len(dates)} stocks={len(stocks)}", flush=True)
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_jsonl.open("w", encoding="utf-8") as mf:
            for i, row in enumerate(parse_ok):
                fid = str(row.get("factor_id"))
                formula = str(row.get("formula_v9") or "").strip()
                try:
                    m = precompute_one(
                        eng,
                        formula,
                        dates=dates,
                        stocks=stocks,
                        y_mat=y_mat,
                        label_days=args.label_days,
                    )
                    m["factor_id"] = fid
                    m["formula"] = formula
                    metrics_by_id[fid] = m
                    mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception as exc:
                    err = {
                        "factor_id": fid,
                        "formula": formula,
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    exec_fail.append(err)
                    mf.write(json.dumps(err, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(parse_ok):
                    print(
                        f"[precompute] {i+1}/{len(parse_ok)} "
                        f"pass={len(metrics_by_id)} fail={len(exec_fail)}",
                        flush=True,
                    )
    else:
        print("[precompute] skipped")

    entries: list[dict[str, Any]] = []
    kept_ids: list[str] = []
    for row in parse_ok:
        fid = str(row.get("factor_id"))
        formula = str(row.get("formula_v9") or "").strip()
        m = metrics_by_id.get(fid)
        if not args.skip_precompute:
            if m is None:
                continue
            if float(m.get("finite_ratio") or 0) < args.min_finite_ratio:
                continue
            if m.get("ic") is None:
                # 仍可进库，但标记无 IC（截面不足等）
                pass
        item: dict[str, Any] = {
            "factor_id": fid,
            "expr": formula,
            "topic": _topic(row),
            "description": _desc(row),
            "operators": _ops_of(row) or list(extract_dsl_operator_names(formula)),
            "family": row.get("family"),
            "source_library": row.get("source_library"),
        }
        if m:
            item["metrics"] = {
                "ic": m.get("ic"),
                "icir": m.get("icir"),
                "abs_ic": m.get("abs_ic"),
                "finite_ratio": m.get("finite_ratio"),
                "n_unique": m.get("n_unique"),
                "label_days": m.get("label_days"),
                "window": {"start": args.start, "end": args.end},
            }
        entries.append(item)
        kept_ids.append(fid)

    export_yaml(
        entries,
        args.out_yaml,
        schema_note=f"production_default_core_v9 A-share; precompute={not args.skip_precompute}",
        values_precomputed=not args.skip_precompute,
    )

    summary = {
        "library_json": str(args.library_json),
        "candidates": len(rows),
        "parse_ok": len(parse_ok),
        "parse_fail": len(parse_fail),
        "exec_pass": len(metrics_by_id),
        "exec_fail": len(exec_fail),
        "exported": len(entries),
        "out_yaml": str(args.out_yaml),
        "precompute_window": {"start": args.start, "end": args.end, "label_days": args.label_days},
        "abs_ic_quantiles": None,
        "parse_fail_samples": parse_fail[:20],
        "exec_fail_samples": exec_fail[:20],
    }
    abs_ics = [float(m["abs_ic"]) for m in metrics_by_id.values() if m.get("abs_ic") is not None]
    if abs_ics:
        arr = np.asarray(sorted(abs_ics))
        summary["abs_ic_quantiles"] = {
            "p50": float(np.quantile(arr, 0.5)),
            "p90": float(np.quantile(arr, 0.9)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("candidates", "parse_ok", "exec_pass", "exec_fail", "exported")}, ensure_ascii=False))
    print(f"[done] yaml -> {args.out_yaml}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
