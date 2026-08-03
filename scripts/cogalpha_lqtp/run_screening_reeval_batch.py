#!/usr/bin/env python3
"""Materialize + extended-eval all screening manifest factors (LQTP → FE → Python)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
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

from api.mining_integration import validate_factor_engine_dsl  # noqa: E402

from scripts.cogalpha_lqtp.ast_translator import dsl_to_lqtp, lqtp_to_fe_dsl  # noqa: E402
from scripts.cogalpha_lqtp.data_access_panel import ashare_materialize_data_source_config  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    LqtpTokenManager,
    factor_values_to_long_df,
    run_factor_formula,
)
from scripts.cogalpha_lqtp.lqtp_converter import convert_python  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_dsl_compat import (  # noqa: E402
    eval_route_for_entry,
    is_lqtp_native_dsl,
)
from scripts.cogalpha_lqtp.materialize import (  # noqa: E402
    _normalize_dsl_for_fe,
    materialize_factor,
)
from scripts.cogalpha_lqtp.memory_utils import release_memory  # noqa: E402
from scripts.cogalpha_lqtp.python_materialize import materialize_python_factor  # noqa: E402

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-06-30"
TARGET_FACTOR_COUNT = 171


def _log(msg: str) -> None:
    print(msg, flush=True)


def _date_int(s: str) -> int:
    return int(s.replace("-", ""))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT_BATCH_PARSED = ROOT / "data/cogalpha_lqtp_batch/parsed_factors.json"


def _python_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for rec in _load_json(path):
        code = rec.get("python_code", "")
        if code.strip():
            out[rec["function_name"]] = code
    return out


def _load_python_fallbacks(work: Path) -> dict[str, str]:
    """Merge screening parsed + cogalpha batch python for materialize fallback."""
    out = _python_map(work / "screening_reeval_parsed_factors.json")
    if DEFAULT_BATCH_PARSED.exists():
        for rec in _load_json(DEFAULT_BATCH_PARSED):
            code = rec.get("python_code", "")
            if code.strip():
                out.setdefault(rec["function_name"], code)
    return out


def _lqtp_creds() -> tuple[str, str, str] | None:
    user = os.environ.get("LQTP_USERNAME", "").strip()
    pwd = os.environ.get("LQTP_PASSWORD", "").strip()
    server = os.environ.get("LQTP_SERVER", "lqtp.example.com").strip()
    if user and pwd:
        return user, pwd, server
    return None


def _save_long_df(out_path: Path, long_df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = long_df.copy()
    if "trade_date" not in work.columns:
        if "datetime" in work.columns:
            dt = pd.to_datetime(work["datetime"])
            work["trade_date"] = dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day
        elif "date" in work.columns:
            dt = pd.to_datetime(work["date"])
            work["trade_date"] = dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day
    if "symbol" not in work.columns and "asset" in work.columns:
        work["symbol"] = work["asset"].astype(str)
    work = work[["trade_date", "symbol", "value"]].dropna(subset=["value"])
    work["trade_date"] = work["trade_date"].astype(int)
    work["symbol"] = work["symbol"].astype(str)
    # Drop ±inf / non-finite values so DuckDB RankIC doesn't collapse
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work[np.isfinite(work["value"].to_numpy(dtype="float64", copy=False))]
    if work.empty:
        raise RuntimeError(f"{out_path.parent.name}: no finite factor values after sanitize")
    work.to_parquet(out_path, index=False)


def _prepare_entry_dsl(entry: dict[str, Any], python_code: str = "") -> str:
    """Normalize pack DSL; if missing, try Python→LQTP conversion. Updates entry in place."""
    name = entry["function_name"]
    raw = (entry.get("lqtp_formula") or entry.get("dsl") or "").strip()
    if not raw and python_code.strip():
        conv = convert_python(python_code, name=name, validate_fe=False)
        cand = (conv.lqtp_formula or "").strip()
        # Reject residual pandas leftovers
        if cand and not any(tok in cand for tok in (".rolling", ".pct_change", ".groupby", "min_pe")):
            raw = cand
    if not raw:
        return ""
    formula = _normalize_dsl_for_fe(raw, factor_name=name)
    if not formula:
        return ""
    entry["dsl"] = formula
    entry["lqtp_formula"] = formula
    # Recompute route after normalization / conversion
    if is_lqtp_native_dsl(formula):
        entry["status"] = "ready"
        entry["eval_route"] = "lqtp_dsl"
        entry["lqtp_native"] = True
    else:
        fe = lqtp_to_fe_dsl(formula)
        ok, _ = validate_factor_engine_dsl(fe, surface="compat")
        ok2, _ = validate_factor_engine_dsl(formula, surface="compat")
        if ok or ok2:
            entry["status"] = "ready"
            entry["eval_route"] = eval_route_for_entry(status="ready", dsl=formula)
            entry["lqtp_native"] = is_lqtp_native_dsl(formula)
        elif not (entry.get("dsl") or "").strip() and python_code.strip():
            entry["status"] = "python"
            entry["eval_route"] = "local_python"
            entry["lqtp_native"] = False
    return formula


def _try_lqtp_materialize(
    *,
    entry: dict[str, Any],
    out_path: Path,
    start: str,
    end: str,
    creds: tuple[str, str, str],
    fwd_path: Path | None,
) -> tuple[Path, str, str] | None:
    user, pwd, server = creds
    name = entry["function_name"]
    formula = _normalize_dsl_for_fe(
        (entry.get("lqtp_formula") or entry.get("dsl") or "").strip(),
        factor_name=name,
    )
    if not formula:
        return None
    if not is_lqtp_native_dsl(formula):
        converted = _normalize_dsl_for_fe(
            dsl_to_lqtp(lqtp_to_fe_dsl(formula)),
            factor_name=name,
        )
        if is_lqtp_native_dsl(converted):
            formula = converted
        else:
            return None
    # LqtpTokenManager.login(server, username, password)
    auth = LqtpTokenManager.login(
        server,
        user,
        pwd,
        refresh_interval_seconds=int(os.environ.get("LQTP_TOKEN_REFRESH_SECONDS", "1500")),
        login_retries=int(os.environ.get("LQTP_LOGIN_RETRIES", "3")),
        login_wait_sec=float(os.environ.get("LQTP_LOGIN_WAIT_SEC", "5")),
    )
    auth.maybe_refresh()
    # Values-only: skip platform analyze and local IC (eval phase handles RankIC).
    _ = fwd_path  # reserved for future universe alignment
    resp = run_factor_formula(
        token=auth.token,
        formula=formula,
        begin_date=_date_int(start),
        end_date=_date_int(end),
        warmup=1,
        analyze=False,
        server=server,
        factor_name="",
    )
    if resp.error:
        raise RuntimeError(resp.error)
    daily_values = list(resp.values)
    if not daily_values:
        raise RuntimeError("RunFactor returned no factor values")
    long_df = factor_values_to_long_df(daily_values)
    _save_long_df(out_path, long_df)
    release_memory(long_df, daily_values)
    entry["dsl"] = formula
    entry["lqtp_formula"] = formula
    entry["eval_route"] = "lqtp_dsl"
    entry["lqtp_native"] = True
    return out_path, "lqtp_dsl", formula


def materialize_with_priority(
    *,
    entry: dict[str, Any],
    python_code: str,
    data_cfg: dict[str, Any],
    lake_root: Path,
    start: str,
    end: str,
    symbol_chunk: int,
    python_workers: int,
    force: bool,
    creds: tuple[str, str, str] | None,
    fwd_path: Path | None,
) -> tuple[Path, str, str]:
    name = entry["function_name"]
    out_path = lake_root / name / "values.parquet"
    if out_path.exists() and not force:
        engine = entry.get("materialized_via") or (
            "python" if entry.get("status") == "python" else "factor_engine"
        )
        if entry.get("eval_route") == "lqtp_dsl" and entry.get("materialized_via") == "lqtp_dsl":
            engine = "lqtp_dsl"
        dsl = entry.get("dsl") or "(python only)"
        return out_path, engine, dsl

    # Prefer Python when catalog says so (before DSL prep, which may rewrite eval_route).
    prefer_python = entry.get("eval_route") == "local_python" and bool(python_code.strip())
    if prefer_python:
        out = materialize_python_factor(
            function_name=name,
            python_code=python_code,
            start=start,
            end=end,
            lake_root=lake_root,
            symbol_chunk=symbol_chunk,
            workers=python_workers,
        )
        entry["materialized_via"] = "python"
        entry["eval_route"] = "local_python"
        return out, "python", (entry.get("dsl") or "(python only)")

    dsl = _prepare_entry_dsl(entry, python_code)

    # 1) LQTP RunFactor when formula is native after normalize / python→dsl
    if creds is not None:
        try:
            hit = _try_lqtp_materialize(
                entry=entry,
                out_path=out_path,
                start=start,
                end=end,
                creds=creds,
                fwd_path=fwd_path,
            )
            if hit is not None:
                entry["materialized_via"] = "lqtp_dsl"
                return hit
        except Exception as exc:  # noqa: BLE001
            print(f"  LQTP materialize fallback {name}: {exc}", flush=True)

    # 2) factor_engine
    if dsl:
        try:
            out = materialize_factor(
                factor_id=name,
                dsl=dsl,
                data_source_cfg=data_cfg,
                lake_root=lake_root,
            )
            entry["materialized_via"] = "factor_engine"
            entry["status"] = "ready"
            return out, "factor_engine", dsl
        except Exception as exc:  # noqa: BLE001
            print(f"  factor_engine fallback {name}: {exc}", flush=True)

    # 3) Python payload
    if not python_code.strip():
        raise RuntimeError(f"{name}: no LQTP/FE path and missing python_code")
    out = materialize_python_factor(
        function_name=name,
        python_code=python_code,
        start=start,
        end=end,
        lake_root=lake_root,
        symbol_chunk=symbol_chunk,
        workers=python_workers,
    )
    entry["materialized_via"] = "python"
    return out, "python", dsl or "(python only)"


def run_materialize_phase(
    *,
    work: Path,
    catalog: list[dict[str, Any]],
    python_map: dict[str, str],
    start: str,
    end: str,
    only: set[str] | None,
    force: bool,
    symbol_chunk: int,
    python_workers: int,
) -> dict[str, Any]:
    lake = work / "factor_lake"
    progress_path = work / "screening_reeval_materialize.json"
    progress = _load_json(progress_path) if progress_path.exists() else {"completed": [], "failed": {}}
    completed = set(progress.get("completed", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))

    data_cfg = ashare_materialize_data_source_config(start_date=start, end_date=end)
    creds = _lqtp_creds()
    fwd_path = work / "lqtp_fwd_vwap_returns_cache.parquet"
    if not fwd_path.exists():
        fwd_path = work / "lqtp_fwd_close_returns_cache.parquet"
    if not fwd_path.exists():
        fwd_path = None

    names = [e["function_name"] for e in catalog]
    if only:
        names = [n for n in names if n in only]

    for i, entry in enumerate(catalog, 1):
        name = entry["function_name"]
        if name not in names:
            continue
        if name in completed and not force:
            continue
        py = python_map.get(name, "")
        print(f"[mat {i}/{len(names)}] {name} route={entry.get('eval_route')} status={entry.get('status')}", flush=True)
        try:
            out_path, engine, dsl = materialize_with_priority(
                entry=entry,
                python_code=py,
                data_cfg=data_cfg,
                lake_root=lake,
                start=start,
                end=end,
                symbol_chunk=symbol_chunk,
                python_workers=python_workers,
                force=force,
                creds=creds,
                fwd_path=fwd_path,
            )
            progress.setdefault("details", {})[name] = {
                "engine": engine,
                "dsl": dsl[:200],
                "path": str(out_path),
                "at": datetime.now().isoformat(),
            }
            completed.add(name)
            failed.pop(name, None)
        except Exception as exc:  # noqa: BLE001
            failed[name] = str(exc)
            print(f"  FAIL {name}: {exc}", flush=True)
            traceback.print_exc()
        progress["completed"] = sorted(completed)
        progress["failed"] = failed
        _save_json(progress_path, progress)

    _save_json(work / "screening_reeval_catalog.json", catalog)
    return progress


def run_eval_phase(
    *,
    work: Path,
    start: str,
    end: str,
    only: set[str] | None,
    workers: int,
    force_aux_cache: bool,
    skip_lookahead: bool = True,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.cogalpha_lqtp.eval_lake_fast",
        "--work-dir",
        str(work),
        "--start",
        start,
        "--end",
        end,
        "--returns-source",
        "data_access",
        "--return-kind",
        "vwap",
        "--workers",
        str(workers),
        "--catalog",
        str(work / "screening_reeval_catalog.json"),
        "--parsed-json",
        str(work / "screening_reeval_parsed_factors.json"),
        "--progress-file",
        str(work / "screening_reeval_progress.json"),
        "--report-subdir",
        "reports_screening_reeval",
    ]
    if force_aux_cache:
        cmd.append("--force-aux-cache")
    if only:
        cmd.extend(["--only", *sorted(only)])
    cmd.append("--skip-lookahead" if skip_lookahead else "--no-skip-lookahead")
    print("eval:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screening manifest re-eval batch")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--build-catalog", action="store_true", help="Rebuild catalog from manifest first")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force-materialize", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run only factors in materialize failed map")
    parser.add_argument("--force-aux-cache", action="store_true")
    parser.add_argument("--symbol-chunk", type=int, default=400)
    parser.add_argument("--python-workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="eval_lake_fast workers")
    parser.add_argument(
        "--skip-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Defer LOOKAHEAD_DEFERRED_FACTORS (default). Pass --no-skip-lookahead to include them.",
    )
    args = parser.parse_args()

    work = args.work_dir

    if args.build_catalog or not (work / "screening_reeval_catalog.json").exists():
        rc = subprocess.call(
            [sys.executable, "-m", "scripts.cogalpha_lqtp.build_screening_reeval_catalog", "--work-dir", str(work)]
        )
        if rc != 0:
            return rc

    catalog = _load_json(work / "screening_reeval_catalog.json")
    python_map = _load_python_fallbacks(work)
    target = len(catalog)

    mat_path = work / "screening_reeval_materialize.json"
    mat_progress = _load_json(mat_path) if mat_path.exists() else {"completed": [], "failed": {}}

    only = set(args.only) if args.only else None
    if args.retry_failed:
        failed_names = sorted(mat_progress.get("failed", {}).keys())
        if not failed_names:
            _log("retry-failed: nothing in failed map")
            return 0
        only = set(failed_names)
        args.force_materialize = True
        mat_progress["failed"] = {}
        _save_json(mat_path, mat_progress)
        done_path = work / "screening_reeval_DONE.json"
        if done_path.exists():
            done_path.unlink()
        _log(f"retry-failed: {len(only)} factors")

    mat_done = len(mat_progress.get("completed", []))
    skip_mat = (
        not args.eval_only
        and not args.force_materialize
        and mat_done >= target
    )

    if not args.eval_only and not skip_mat:
        mat = run_materialize_phase(
            work=work,
            catalog=catalog,
            python_map=python_map,
            start=args.start,
            end=args.end,
            only=only,
            force=args.force_materialize,
            symbol_chunk=args.symbol_chunk,
            python_workers=args.python_workers,
        )
        mat_done = len(mat.get("completed", []))
        print(
            f"materialize done ok={mat_done} fail={len(mat.get('failed', {}))}",
            flush=True,
        )
        if args.materialize_only:
            return 0 if not mat.get("failed") else 1
    elif skip_mat:
        _log(f"skip materialize ({mat_done}/{target} already done)")

    rc = run_eval_phase(
        work=work,
        start=args.start,
        end=args.end,
        only=only,
        workers=args.workers,
        force_aux_cache=args.force_aux_cache,
        skip_lookahead=args.skip_lookahead,
    )
    if rc == 0:
        subprocess.call([sys.executable, "-m", "scripts.cogalpha_lqtp.render_rankic_screening_index"])
        subprocess.call(
            [
                sys.executable,
                "-m",
                "scripts.cogalpha_lqtp.compute_factor_corr_matrix",
                "--work-dir",
                str(work),
                "--sample-days",
                "80",
            ]
        )
        done_path = work / "screening_reeval_DONE.json"
        prog = _load_json(work / "screening_reeval_progress.json") if (work / "screening_reeval_progress.json").exists() else {}
        _save_json(
            done_path,
            {
                "finished_at": datetime.now().isoformat(),
                "materialized": mat_done,
                "evaluated": len(prog.get("completed", [])),
                "failed_eval": prog.get("failed", {}),
            },
        )
        _log(f"pipeline complete → {done_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
