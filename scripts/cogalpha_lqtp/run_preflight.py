#!/usr/bin/env python3
"""Preflight gate: verify factors before bulk materialize / LQTP eval.

Checks per factor:
  1. catalog + DSL syntax (factor_engine)
  2. local materialize on smoke data
  3. value sanity (rows, finite ratio, dispersion, date coverage)
  4. Python reference vs factor_engine DSL (rank correlation)
  5. LQTP upload eval + backtest (optional, --skip-lqtp)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import validate_factor_engine_dsl  # noqa: E402

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    evaluate_factor,
    factor_values_to_long_df,
    fetch_lqtp_universe,
    login,
    long_df_to_daily_values,
    run_backtest_from_weights,
    top_quantile_weights,
)
from scripts.cogalpha_lqtp.materialize import _ashare_smoke_data_source, materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import DEFAULT_FACTORS, _run_py, _yyyymmdd  # noqa: E402

PREFLIGHT_FACTORS = DEFAULT_FACTORS

# factor_engine ema uses adjust=False; original CogAlpha Python often uses pandas default adjust=True.
PYTHON_REFERENCE_OVERRIDES: dict[str, str] = {
    "factor_persistence_ewma": """
def factor_persistence_ewma(df):
    returns = df["close"].pct_change()
    abs_return_diff = returns.diff().abs()
    smoothness = -abs_return_diff.ewm(span=10, min_periods=1, adjust=False).mean()
    return smoothness.fillna(0)
""",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class FactorPreflight:
    function_name: str
    dsl: str = ""
    passed: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    compare: dict[str, Any] = field(default_factory=dict)
    materialize: dict[str, Any] = field(default_factory=dict)
    lqtp: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))

    def finalize(self) -> None:
        self.passed = all(c.ok for c in self.checks)


def _load_catalog_map(catalog_path: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {item["function_name"]: item for item in catalog}


def _load_python_map(parsed_path: Path) -> dict[str, str]:
    records = json.loads(parsed_path.read_text(encoding="utf-8"))
    return {r["function_name"]: r["python_code"] for r in records}


def _smoke_parquet_path(smoke_dir: Path) -> Path:
    path = smoke_dir / "smoke_2024Q1.parquet"
    if not path.exists():
        paths = sorted(smoke_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no parquet under {smoke_dir}")
        path = paths[0]
    return path


def _symbol_panel(parquet_path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_parquet(parquet_path)
    colmap = {
        "TradeDate": "date",
        "Symbol": "symbol",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "PreClose": "pre_close",
        "Volume": "volume",
        "Amount": "amount",
        "Return": "return",
        "Vwap": "vwap",
    }
    work = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns})
    work["date"] = pd.to_datetime(work["date"])
    panels: dict[str, pd.DataFrame] = {}
    for symbol, group in work.groupby("symbol", sort=True):
        frame = group.sort_values("date").reset_index(drop=True)
        panels[str(symbol)] = frame
    return panels


def _run_python_factor(code: str, function_name: str, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from scripts.cogalpha_lqtp.python_runtime import build_python_exec_namespace, prepare_factor_dataframe

    namespace = build_python_exec_namespace()
    exec(code, namespace)  # noqa: S102
    if function_name not in namespace:
        raise RuntimeError(f"{function_name} not defined in python_code")
    func = namespace[function_name]
    rows: list[dict[str, Any]] = []
    for symbol, frame in panels.items():
        out = func(prepare_factor_dataframe(frame.copy()))
        if isinstance(out, pd.DataFrame):
            if out.shape[1] != 1:
                raise RuntimeError(f"{function_name} returned DataFrame with {out.shape[1]} cols")
            series = out.iloc[:, 0]
        elif isinstance(out, pd.Series):
            series = out
        else:
            raise RuntimeError(f"{function_name} returned {type(out)}")
        series = series.reindex(frame.index)
        for dt, val in zip(frame["date"], series, strict=False):
            if pd.notna(val) and np.isfinite(val):
                rows.append({"datetime": pd.Timestamp(dt).normalize(), "asset": symbol, "value_py": float(val)})
    return pd.DataFrame(rows)


def _check_value_sanity(long_df: pd.DataFrame, *, min_rows: int, min_symbols: int) -> CheckResult:
    if long_df.empty:
        return CheckResult("value_sanity", False, "empty output")
    n_rows = len(long_df)
    n_syms = long_df["asset"].nunique()
    finite = np.isfinite(long_df["value"].astype(float))
    finite_ratio = float(finite.mean())
    vals = long_df.loc[finite, "value"].astype(float)
    std = float(vals.std()) if len(vals) > 1 else 0.0
    n_dates = long_df["datetime"].nunique()
    ok = (
        n_rows >= min_rows
        and n_syms >= min_symbols
        and finite_ratio >= 0.95
        and std > 1e-12
        and n_dates >= 5
    )
    detail = (
        f"rows={n_rows}, symbols={n_syms}, dates={n_dates}, "
        f"finite={finite_ratio:.3f}, std={std:.6g}"
    )
    return CheckResult("value_sanity", ok, detail)


def _compare_python_vs_dsl(dsl_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[CheckResult, dict[str, Any]]:
    dsl = dsl_df.rename(columns={"value": "value_dsl"}).copy()
    dsl["datetime"] = pd.to_datetime(dsl["datetime"]).dt.normalize()
    py = py_df.copy()
    py["datetime"] = pd.to_datetime(py["datetime"]).dt.normalize()
    merged = dsl.merge(py, on=["datetime", "asset"], how="inner")
    if merged.empty:
        return CheckResult("python_parity", False, "no overlapping rows"), {"overlap_rows": 0}

    x = merged["value_dsl"].astype(float)
    y = merged["value_py"].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    merged = merged.loc[mask]
    if len(merged) < 10:
        return CheckResult("python_parity", False, f"too few overlap rows={len(merged)}"), {}

    pearson = float(x[mask].corr(y[mask]))
    spearman = float(x[mask].rank().corr(y[mask].rank()))

    daily_corr: list[float] = []
    for _, group in merged.groupby("datetime"):
        if len(group) < 3:
            continue
        daily_corr.append(float(group["value_dsl"].rank().corr(group["value_py"].rank())))
    mean_daily = float(np.nanmean(daily_corr)) if daily_corr else float("nan")

    # allow small fillna / warmup differences
    ok = spearman >= 0.98 and (np.isnan(mean_daily) or mean_daily >= 0.95)
    stats = {
        "overlap_rows": int(len(merged)),
        "pearson": pearson,
        "spearman": spearman,
        "mean_daily_rank_corr": mean_daily,
        "max_abs_diff": float((merged["value_dsl"] - merged["value_py"]).abs().max()),
    }
    detail = (
        f"overlap={len(merged)}, spearman={spearman:.4f}, "
        f"daily_rank={mean_daily:.4f}, max_diff={stats['max_abs_diff']:.6g}"
    )
    return CheckResult("python_parity", ok, detail), stats


def preflight_one(
    *,
    function_name: str,
    catalog_entry: dict[str, Any],
    python_code: str | None,
    data_cfg: dict[str, Any],
    lake_root: Path,
    token: str | None,
    begin_i: int,
    end_i: int,
    server: str,
    skip_lqtp: bool,
    min_rows: int,
    min_symbols: int,
    panels: dict[str, pd.DataFrame] | None,
) -> FactorPreflight:
    result = FactorPreflight(function_name=function_name)
    dsl = catalog_entry.get("dsl", "")
    result.dsl = dsl

    if catalog_entry.get("status") != "ready" or not dsl:
        result.add("catalog_ready", False, f"status={catalog_entry.get('status')}")
        result.finalize()
        return result
    result.add("catalog_ready", True, "ready")

    valid, msg = validate_factor_engine_dsl(dsl)
    result.add("dsl_validate", valid, msg if not valid else "ok")
    if not valid:
        result.finalize()
        return result

    try:
        out_path = materialize_factor(
            factor_id=function_name,
            dsl=dsl,
            data_source_cfg=data_cfg,
            lake_root=lake_root,
        )
        long_df = pd.read_parquet(out_path)
        result.materialize = {
            "values_path": str(out_path),
            "rows": int(len(long_df)),
            "symbols": int(long_df["asset"].nunique()),
            "date_min": str(long_df["datetime"].min()),
            "date_max": str(long_df["datetime"].max()),
        }
        result.add("materialize", True, f"rows={len(long_df)}")
    except Exception as exc:  # noqa: BLE001
        result.add("materialize", False, str(exc))
        result.finalize()
        return result

    sanity = _check_value_sanity(long_df, min_rows=min_rows, min_symbols=min_symbols)
    result.add(sanity.name, sanity.ok, sanity.detail)

    if python_code and panels:
        try:
            py_df = _run_python_factor(python_code, function_name, panels)
            cmp_check, stats = _compare_python_vs_dsl(long_df, py_df)
            result.compare = stats
            result.add(cmp_check.name, cmp_check.ok, cmp_check.detail)
        except Exception as exc:  # noqa: BLE001
            result.add("python_parity", False, str(exc))
    else:
        result.add("python_parity", False, "missing python reference")

    if not skip_lqtp and token:
        try:
            daily_values = long_df_to_daily_values(long_df)
            analysis, eval_mode = evaluate_factor(
                token=token,
                daily_values=daily_values,
                begin_date=begin_i,
                end_date=end_i,
                server=server,
            )
            universe = fetch_lqtp_universe(
                token=token,
                begin_date=begin_i,
                end_date=end_i,
                server=server,
            )
            weights = top_quantile_weights(
                factor_values_to_long_df(daily_values),
                allowed_symbols=universe,
            )
            bt_rows, bt_id = run_backtest_from_weights(
                token=token,
                weights=weights,
                begin_date=begin_i,
                end_date=end_i,
                server=server,
            )
            lqtp_ok = (
                analysis.get("mean_ic") is not None
                and len(analysis.get("trade_dates", [])) >= 5
                and len(bt_rows) >= 5
            )
            result.lqtp = {
                "eval_mode": eval_mode,
                "mean_ic": analysis.get("mean_ic"),
                "icir": analysis.get("icir"),
                "coverage": analysis.get("coverage"),
                "backtest_days": len(bt_rows),
                "backtest_id": bt_id,
            }
            result.add(
                "lqtp_eval",
                lqtp_ok,
                f"mode={eval_mode}, ic={analysis.get('mean_ic')}, bt_days={len(bt_rows)}",
            )
        except Exception as exc:  # noqa: BLE001
            result.add("lqtp_eval", False, str(exc))
    elif skip_lqtp:
        result.add("lqtp_eval", True, "skipped")

    result.finalize()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight verify factors before bulk run")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_preflight")
    parser.add_argument("--factors", nargs="*", default=PREFLIGHT_FACTORS)
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--n-symbols", type=int, default=50)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-symbols", type=int, default=20)
    parser.add_argument("--skip-lqtp", action="store_true")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--regen-catalog", action="store_true")
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    parsed = work / "parsed_factors.json"
    catalog_path = work / "dsl_catalog.json"
    smoke_dir = work / "smoke_StockDailyBar"
    lake = work / "factor_lake"
    report_path = work / "preflight_report.json"

    if args.regen_catalog or not catalog_path.exists():
        _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(parsed))
        _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog_path))
    elif not parsed.exists():
        parsed = ROOT / "data/cogalpha_lqtp_batch/parsed_factors.json"
        if not parsed.exists():
            _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(work / "parsed_factors.json"))
            parsed = work / "parsed_factors.json"

    if not catalog_path.exists():
        catalog_path = ROOT / "data/cogalpha_lqtp_batch/dsl_catalog.json"

    if not smoke_dir.exists():
        _run_py(
            SCRIPTS / "smoke_data.py",
            "--out-dir",
            str(smoke_dir),
            "--start",
            args.start,
            "--end",
            args.end,
            "--n-symbols",
            str(args.n_symbols),
        )

    catalog_map = _load_catalog_map(catalog_path)
    python_map = _load_python_map(parsed if parsed.exists() else work / "parsed_factors.json")
    panels = _symbol_panel(_smoke_parquet_path(smoke_dir))
    data_cfg = _ashare_smoke_data_source(
        parquet_root=smoke_dir,
        start_date=args.start,
        end_date=args.end,
    )

    token: str | None = None
    if not args.skip_lqtp:
        auth = login(args.server, args.username, args.password)
        token = auth.access_token

    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)

    results: list[FactorPreflight] = []
    for name in args.factors:
        if name not in catalog_map:
            fp = FactorPreflight(function_name=name)
            fp.add("catalog_ready", False, "not in catalog")
            fp.finalize()
            results.append(fp)
            continue
        print(f"preflight {name} ...")
        py_code = PYTHON_REFERENCE_OVERRIDES.get(name) or python_map.get(name)
        fp = preflight_one(
            function_name=name,
            catalog_entry=catalog_map[name],
            python_code=py_code,
            data_cfg=data_cfg,
            lake_root=lake,
            token=token,
            begin_i=begin_i,
            end_i=end_i,
            server=args.server,
            skip_lqtp=args.skip_lqtp,
            min_rows=args.min_rows,
            min_symbols=args.min_symbols,
            panels=panels,
        )
        results.append(fp)
        status = "PASS" if fp.passed else "FAIL"
        failed = [c.name for c in fp.checks if not c.ok]
        print(f"  {status}" + (f" ({', '.join(failed)})" if failed else ""))

    passed = [r.function_name for r in results if r.passed]
    failed = [r.function_name for r in results if not r.passed]
    payload = {
        "factors": args.factors,
        "date_range": [args.start, args.end],
        "n_symbols": args.n_symbols,
        "all_passed": len(failed) == 0,
        "passed": passed,
        "failed": failed,
        "results": [
            {
                "function_name": r.function_name,
                "dsl": r.dsl,
                "passed": r.passed,
                "checks": [asdict(c) for c in r.checks],
                "compare": r.compare,
                "materialize": r.materialize,
                "lqtp": r.lqtp,
            }
            for r in results
        ],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"all_passed": payload["all_passed"], "passed": passed, "failed": failed}, ensure_ascii=False))
    print(f"report -> {report_path}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
