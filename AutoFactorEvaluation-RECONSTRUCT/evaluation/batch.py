"""Public provider-neutral factor-pack evaluation API and CLI."""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from typing import Sequence

from .batch_models import BatchEvaluationConfig, BatchRunSummary, FactorRunRecord, SplitBoundaries
from .batch_runner import build_synthetic_market_frame, run_factor_pack_evaluation
from .factor_pack import FactorDefinition, FactorPack, FactorPackProvider


def load_factor_pack(provider: str) -> FactorPack:
    module_name, separator, attribute = provider.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("provider must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    pack = factory()
    if not isinstance(pack, FactorPack):
        raise TypeError(f"provider {provider} returned {type(pack).__name__}, expected FactorPack")
    pack.validate()
    return pack


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an external versioned factor pack")
    parser.add_argument("--provider", required=True, help="module:function returning FactorPack")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--universe-id", default="A_SHARE_ALL_A_EX_ST")
    parser.add_argument("--universe-dataset", default=None)
    parser.add_argument("--require-point-in-time-universe", action="store_true")
    parser.add_argument("--backend", default="pandas")
    parser.add_argument("--run-mode", choices=["research", "production"], default="research")
    parser.add_argument("--horizons", default="1,5,21")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--entry-lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", action="append", default=[])
    parser.add_argument("--materialize-staging", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=320)
    parser.add_argument("--synthetic-symbols", type=int, default=8)
    args = parser.parse_args(argv)

    pack = load_factor_pack(args.provider)
    require_universe = args.require_point_in_time_universe or args.run_mode == "production"
    config = BatchEvaluationConfig(
        market=args.market,
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        universe_id=args.universe_id,
        universe_dataset=args.universe_dataset,
        require_point_in_time_universe=require_universe,
        backend=args.backend,
        run_mode=args.run_mode,
        horizons=_parse_csv_ints(args.horizons),
        batch_size=args.batch_size,
        min_assets=args.min_assets,
        n_quantiles=args.n_quantiles,
        entry_lag=args.entry_lag,
        cost_bps=args.cost_bps,
        fdr_alpha=args.fdr_alpha,
        materialize_staging=args.materialize_staging,
        publish=args.publish,
        factor_version=f"{pack.name}.{pack.version}",
        strict=not args.allow_partial,
        resume=not args.no_resume,
        limit=args.limit,
        selected_factors=tuple(args.factor),
    )
    frame = None
    snapshot_id = None
    if args.synthetic:
        frame = build_synthetic_market_frame(
            periods=args.synthetic_periods,
            symbols=args.synthetic_symbols,
        )
        snapshot_id = f"synthetic:{pack.name}"
    try:
        summary = run_factor_pack_evaluation(
            pack,
            config,
            output_dir=args.output_dir,
            market_frame=frame,
            snapshot_id=snapshot_id,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str))
    return 0 if summary.status == "success" else 1


__all__ = [
    "BatchEvaluationConfig",
    "BatchRunSummary",
    "FactorDefinition",
    "FactorPack",
    "FactorPackProvider",
    "FactorRunRecord",
    "SplitBoundaries",
    "build_synthetic_market_frame",
    "load_factor_pack",
    "run_factor_pack_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
