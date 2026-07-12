#!/usr/bin/env python3
"""Materialize CogAlpha DSL factors via factor_engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.dsl_parser import parse_factor  # noqa: E402
from api.factor import Factor  # noqa: E402
from api.mining_integration import (  # noqa: E402
    default_ashare_pv_data_source_config,
    validate_factor_engine_dsl,
)
from backend.factory import build_backend  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from storage.cache import CacheManager  # noqa: E402
from storage.factory import build_data_source  # noqa: E402
from storage.factor_format import series_to_long_table  # noqa: E402


def _ashare_smoke_data_source(
    *,
    parquet_root: Path,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    cfg = default_ashare_pv_data_source_config(
        max_files=10,
        start_date=start_date,
        end_date=end_date,
    )
    cfg["root"] = str(parquet_root)
    return cfg


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


def materialize_factor(
    *,
    factor_id: str,
    dsl: str,
    data_source_cfg: dict[str, Any],
    lake_root: Path,
    allowed_symbols: set[str] | None = None,
) -> Path:
    ok, msg = validate_factor_engine_dsl(dsl)
    if not ok:
        raise RuntimeError(f"DSL invalid for {factor_id}: {msg}")

    data_source = build_data_source(data_source_cfg)
    engine = FactorEngine(
        backend=build_backend("pandas"),
        data_source=data_source,
        cache=CacheManager(),
    )
    factor = parse_factor(dsl, name=factor_id, freq="1d", universe="A_SHARE_ALL_A_EX_ST")
    out = engine.run(factor)
    series = out["result"]
    long_df = series_to_long_table(series)
    long_df["asset"] = long_df["asset"].map(_normalize_symbol)
    long_df = long_df.dropna(subset=["value"])
    if allowed_symbols:
        long_df = long_df[long_df["asset"].isin(allowed_symbols)]
        if long_df.empty:
            raise RuntimeError(f"{factor_id}: no rows after LQTP universe filter")

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
