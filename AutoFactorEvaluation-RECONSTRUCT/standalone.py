#!/usr/bin/env python3
"""Standalone command line entrypoint for AutoFactorEvaluation-RECONSTRUCT.

Examples
--------
python standalone.py doctor
python standalone.py validate "rank(ts_mean(close, 20))" --run-mode production
python standalone.py evaluate-frame "ts_mean(close, 5) / close - 1" market.parquet result.parquet
python standalone.py pipeline --all --ev-workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from platform_bootstrap import activate_platform, platform_diagnostics


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _synthetic_market_frame(periods: int = 12, assets: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    rows: list[dict[str, Any]] = []
    for asset_idx in range(assets):
        asset = f"TEST{asset_idx:03d}"
        for date_idx, date in enumerate(dates):
            close = 100.0 + asset_idx * 7.0 + date_idx * (1.0 + asset_idx * 0.05)
            rows.append(
                {
                    "datetime": date,
                    "asset": asset,
                    "open": close - 0.4,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000.0 + date_idx * 10.0,
                    "amount": close * (1_000.0 + date_idx * 10.0),
                    "ret": close / (close - 1.0) - 1.0,
                    "vwap": close + 0.1,
                }
            )
    return pd.DataFrame(rows)


def command_doctor(args: argparse.Namespace) -> int:
    diagnostics = platform_diagnostics(import_runtime=True)
    if not args.skip_smoke and diagnostics["status"] == "PASS":
        try:
            from integrations.quant_platform import execute_factor_on_frame

            execution = execute_factor_on_frame(
                "rank(ts_mean(close, 3) / close - 1)",
                _synthetic_market_frame(),
                factor_name="standalone_smoke",
                run_mode="research",
            )
            values = execution.result.to_numpy(dtype=float, copy=False)
            diagnostics["smoke"] = {
                "ok": bool(np.isfinite(values).any()),
                "rows": int(len(execution.result)),
                "finite_rows": int(np.isfinite(values).sum()),
                "snapshot_id": execution.snapshot_id,
                "lookback": int(getattr(execution.analysis, "lookback", 0)),
            }
        except Exception as exc:  # pragma: no cover - deployment diagnostic
            diagnostics["smoke"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not diagnostics["smoke"]["ok"]:
            diagnostics["status"] = "FAIL"
    _print(diagnostics)
    return 0 if diagnostics["status"] == "PASS" else 1


def command_validate(args: argparse.Namespace) -> int:
    activate_platform()
    from integrations.quant_platform import validate_factor_formula

    factor, plan, analysis = validate_factor_formula(
        args.formula,
        name=args.name,
        freq=args.freq,
        universe=args.universe,
        backend=args.backend,
        run_mode=args.run_mode,
    )
    payload = {
        "status": "PASS",
        "factor": {
            "name": getattr(factor, "name", args.name),
            "formula": args.formula,
            "freq": args.freq,
            "universe": args.universe,
        },
        "analysis": {
            "lookback": int(getattr(analysis, "lookback", 0)),
            "has_ts_op": bool(getattr(analysis, "has_ts_op", False)),
            "has_cs_op": bool(getattr(analysis, "has_cs_op", False)),
            "referenced_columns": sorted(getattr(analysis, "referenced_columns", set())),
        },
        "plan": repr(plan),
        "backend": args.backend,
        "run_mode": args.run_mode,
    }
    _print(payload)
    return 0


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path)
    raise ValueError(f"unsupported input format: {path.suffix}; use parquet/csv/feather")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
    elif suffix in {".feather", ".arrow"}:
        frame.to_feather(path)
    else:
        raise ValueError(f"unsupported output format: {path.suffix}; use parquet/csv/feather")


def command_evaluate_frame(args: argparse.Namespace) -> int:
    activate_platform()
    from integrations.quant_platform import execute_factor_on_frame

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    frame = _read_frame(input_path)
    execution = execute_factor_on_frame(
        args.formula,
        frame,
        factor_name=args.name,
        backend=args.backend,
        run_mode=args.run_mode,
        time_column=args.time_column,
        instrument_column=args.instrument_column,
    )
    output = execution.result.rename("value").reset_index()
    output.columns = ["datetime", "asset", "value"]
    output["factor_name"] = args.name
    output["data_snapshot_id"] = execution.snapshot_id
    _write_frame(output, output_path)
    _print(
        {
            "status": "PASS",
            "input": str(input_path),
            "output": str(output_path),
            "rows": int(len(output)),
            "finite_rows": int(np.isfinite(output["value"].to_numpy(dtype=float)).sum()),
            "lookback": int(getattr(execution.analysis, "lookback", 0)),
            "snapshot_id": execution.snapshot_id,
        }
    )
    return 0


def command_pipeline(args: argparse.Namespace) -> int:
    activate_platform()
    import pipeline

    sys.argv = ["pipeline.py", *args.pipeline_args]
    result = pipeline.main()
    return int(result or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="validate bundled platform and run a synthetic factor")
    doctor.add_argument("--skip-smoke", action="store_true")
    doctor.set_defaults(func=command_doctor)

    validate = sub.add_parser("validate", help="compile a FactorEngine DSL formula")
    validate.add_argument("formula")
    validate.add_argument("--name", default="standalone_validation")
    validate.add_argument("--freq", default="1d")
    validate.add_argument("--universe", default=None)
    validate.add_argument("--backend", default="pandas")
    validate.add_argument("--run-mode", choices=["research", "production"], default="research")
    validate.set_defaults(func=command_validate)

    evaluate = sub.add_parser("evaluate-frame", help="evaluate a DSL formula on a local frame")
    evaluate.add_argument("formula")
    evaluate.add_argument("input")
    evaluate.add_argument("output")
    evaluate.add_argument("--name", default="standalone_factor")
    evaluate.add_argument("--backend", default="pandas")
    evaluate.add_argument("--run-mode", choices=["research", "production"], default="research")
    evaluate.add_argument("--time-column", default=None)
    evaluate.add_argument("--instrument-column", default=None)
    evaluate.set_defaults(func=command_evaluate_frame)

    pipeline_parser = sub.add_parser("pipeline", help="forward arguments to the existing pipeline")
    pipeline_parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    pipeline_parser.set_defaults(func=command_pipeline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
