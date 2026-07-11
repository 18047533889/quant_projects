from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
PIPELINE = PROJECT / "pipeline.py"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"
DATASETS = ROOT / "data_access" / "config" / "datasets.yaml"


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_pipeline() -> None:
    replace_exact(
        PIPELINE,
        '''    min_assets: int = 20,
    limit: int | None = None,
    synthetic: bool = False,
''',
        '''    min_assets: int = 20,
    entry_lag: int = 1,
    cost_bps: float = 10.0,
    universe_dataset: str | None = None,
    universe_membership_field: str = "is_member",
    tradability_field: str | None = "is_tradable",
    require_point_in_time_universe: bool = False,
    limit: int | None = None,
    synthetic: bool = False,
''',
        "pipeline function arguments",
    )
    replace_exact(
        PIPELINE,
        '''        min_assets=min_assets,
        limit=limit,
        materialize_staging=materialize_staging,
''',
        '''        min_assets=min_assets,
        entry_lag=entry_lag,
        cost_bps=cost_bps,
        universe_dataset=universe_dataset,
        universe_membership_field=universe_membership_field,
        tradability_field=tradability_field,
        require_point_in_time_universe=require_point_in_time_universe,
        limit=limit,
        materialize_staging=materialize_staging,
''',
        "pipeline evaluator config",
    )
    replace_exact(
        PIPELINE,
        '''    p.add_argument("--gtja-min-assets", type=int, default=20)
    p.add_argument("--gtja-limit", type=int, default=None)
''',
        '''    p.add_argument("--gtja-min-assets", type=int, default=20)
    p.add_argument("--gtja-entry-lag", type=int, default=1)
    p.add_argument("--gtja-cost-bps", type=float, default=10.0)
    p.add_argument("--gtja-universe-dataset", default=None)
    p.add_argument("--gtja-universe-membership-field", default="is_member")
    p.add_argument("--gtja-tradability-field", default="is_tradable")
    p.add_argument("--gtja-require-pit-universe", action="store_true")
    p.add_argument("--gtja-limit", type=int, default=None)
''',
        "pipeline CLI arguments",
    )
    replace_exact(
        PIPELINE,
        '''            min_assets=args.gtja_min_assets,
            limit=args.gtja_limit,
            synthetic=args.gtja_synthetic,
''',
        '''            min_assets=args.gtja_min_assets,
            entry_lag=args.gtja_entry_lag,
            cost_bps=args.gtja_cost_bps,
            universe_dataset=args.gtja_universe_dataset,
            universe_membership_field=args.gtja_universe_membership_field,
            tradability_field=args.gtja_tradability_field or None,
            require_point_in_time_universe=args.gtja_require_pit_universe,
            limit=args.gtja_limit,
            synthetic=args.gtja_synthetic,
''',
        "pipeline CLI invocation",
    )


def remove_obsolete_split_helpers() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    start = text.find("def _split_mask(")
    if start >= 0:
        end = text.find("\n\ndef _frame_source", start)
        if end < 0:
            raise RuntimeError("_split_mask end anchor not found")
        text = text[:start] + text[end + 2 :]
    start = text.find("def _subset_by_split(")
    if start >= 0:
        end = text.find("\n\ndef _route_factor", start)
        if end < 0:
            raise RuntimeError("_subset_by_split end anchor not found")
        text = text[:start] + text[end + 2 :]
    EVALUATOR.write_text(text, encoding="utf-8")


def fix_universe_test() -> None:
    replace_exact(
        TESTS,
        '''    assert filtered[["datetime", "asset"]].to_records(index=False).tolist() == [
        (np.datetime64("2024-01-02T00:00:00.000000000"), "A"),
        (np.datetime64("2024-01-03T00:00:00.000000000"), "B"),
    ]
''',
        '''    assert filtered["asset"].tolist() == ["A", "B"]
    assert filtered["datetime"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
''',
        "portable universe test",
    )


def register_universe_dataset() -> None:
    text = DATASETS.read_text(encoding="utf-8")
    if "ashare_universe_daily:" in text:
        return
    anchor = '''ashare_stock_status:
  <<: *ashare_lqtp_defaults
  root: ${ASHARE_PARQUET_ROOT:-/home/shw/quant_projects/data/a_share/lqtp_data}/StockStatus
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    ListedState: string

'''
    addition = anchor + '''# Point-in-time research universe and tradability mask used by production factor evaluation.
# This is a curated clean-data contract, not a current-security-master snapshot. Each row
# must describe what was knowable and tradable on TradeDate; missing rows mean ineligible.
ashare_universe_daily:
  kind: static
  access_mode: published
  layout: hive
  root: ${ASHARE_CLEAN_ROOT:-/home/shw/quant_projects/data/a_share/clean_data}/universe_daily
  glob: "date=*/*.parquet"
  partition_columns: [date]
  time_column: TradeDate
  instrument_column: Symbol
  hive_partitioning: true
  union_by_name: true
  query_policy:
    require_explicit_columns: true
    max_rows: 20_000_000
  schema:
    TradeDate: date
    Symbol: string
    is_member: bool
    is_tradable: bool
    universe_id: string
    membership_reason: string
    exclusion_reason: string

'''
    if anchor not in text:
        raise RuntimeError("ashare_stock_status registry anchor not found")
    DATASETS.write_text(text.replace(anchor, addition, 1), encoding="utf-8")


def main() -> None:
    patch_pipeline()
    remove_obsolete_split_helpers()
    fix_universe_test()
    register_universe_dataset()
    print("GTJA185 pipeline CLI and PIT universe registry finalized")


if __name__ == "__main__":
    main()
