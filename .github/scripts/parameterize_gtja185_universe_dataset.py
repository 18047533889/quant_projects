from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
DATASETS = ROOT / "data_access" / "config" / "datasets.yaml"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
PIPELINE = PROJECT / "pipeline.py"
DOCS = PROJECT / "docs" / "gtja185_batch_evaluation.md"


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_registry() -> None:
    replace_exact(
        DATASETS,
        '''ashare_universe_daily:
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
''',
        '''ashare_universe_daily:
  kind: parametric
  access_mode: published
  layout: hive
  root_template: ${ASHARE_CLEAN_ROOT:-/home/shw/quant_projects/data/a_share/clean_data}/universe_daily/universe_id={universe_id}
  glob_template: "date=*/*.parquet"
  params_schema:
    universe_id:
      type: str
      pattern: "^[A-Za-z0-9_.-]+$"
      path_segment: true
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
''',
        "parameterized universe registry",
    )


def patch_evaluator() -> None:
    replace_exact(
        EVALUATOR,
        '''    result = store.read_result(
        config.universe_dataset,
        columns=list(dict.fromkeys(columns)),
        time_range=(config.start_date, config.end_date)
        if config.start_date or config.end_date
        else None,
        instrument_filter=list(config.instrument_filter)
        if config.instrument_filter
        else None,
    )
''',
        '''    read_params: dict[str, Any] = {}
    parameter_names = set(getattr(dataset, "params_schema", {}) or {})
    if "universe_id" in parameter_names:
        if not str(config.universe_id or "").strip():
            raise ValueError("parameterized universe dataset requires universe_id")
        read_params["universe_id"] = str(config.universe_id)
    elif parameter_names:
        raise ValueError(
            f"universe dataset {config.universe_dataset} has unsupported parameters: "
            f"{sorted(parameter_names)}"
        )
    result = store.read_result(
        config.universe_dataset,
        columns=list(dict.fromkeys(columns)),
        time_range=(config.start_date, config.end_date)
        if config.start_date or config.end_date
        else None,
        instrument_filter=list(config.instrument_filter)
        if config.instrument_filter
        else None,
        **read_params,
    )
''',
        "universe DataAccess parameters",
    )


def patch_pipeline() -> None:
    replace_exact(
        PIPELINE,
        '''    output_dir: str | Path,
    start_date: str | None = None,
''',
        '''    output_dir: str | Path,
    dataset: str | None = None,
    run_mode: str = "research",
    universe_id: str = "A_SHARE_ALL_A_EX_ST",
    start_date: str | None = None,
''',
        "pipeline dataset/run mode arguments",
    )
    replace_exact(
        PIPELINE,
        '''    require_point_in_time_universe: bool = False,
    limit: int | None = None,
''',
        '''    require_point_in_time_universe: bool = False,
    fdr_alpha: float = 0.10,
    limit: int | None = None,
''',
        "pipeline FDR argument",
    )
    replace_exact(
        PIPELINE,
        '''    config = BatchEvaluationConfig(
        market="ashare",
        start_date=start_date,
''',
        '''    resolved_universe_dataset = universe_dataset
    if require_point_in_time_universe and not resolved_universe_dataset:
        resolved_universe_dataset = "ashare_universe_daily"
    config = BatchEvaluationConfig(
        market="ashare",
        dataset=dataset,
        run_mode=run_mode,
        universe_id=universe_id,
        start_date=start_date,
''',
        "pipeline base config",
    )
    replace_exact(
        PIPELINE,
        '''        cost_bps=cost_bps,
        universe_dataset=universe_dataset,
''',
        '''        cost_bps=cost_bps,
        fdr_alpha=fdr_alpha,
        universe_dataset=resolved_universe_dataset,
''',
        "pipeline FDR and universe dataset config",
    )
    replace_exact(
        PIPELINE,
        '''    p.add_argument("--gtja-output", default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185")
    p.add_argument("--gtja-start-date", default=None)
''',
        '''    p.add_argument("--gtja-output", default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185")
    p.add_argument("--gtja-dataset", default=None)
    p.add_argument("--gtja-run-mode", choices=["research", "production"], default="research")
    p.add_argument("--gtja-universe-id", default="A_SHARE_ALL_A_EX_ST")
    p.add_argument("--gtja-start-date", default=None)
''',
        "pipeline dataset/run mode CLI",
    )
    replace_exact(
        PIPELINE,
        '''    p.add_argument("--gtja-cost-bps", type=float, default=10.0)
    p.add_argument("--gtja-universe-dataset", default=None)
''',
        '''    p.add_argument("--gtja-cost-bps", type=float, default=10.0)
    p.add_argument("--gtja-fdr-alpha", type=float, default=0.10)
    p.add_argument("--gtja-universe-dataset", default=None)
''',
        "pipeline FDR CLI",
    )
    replace_exact(
        PIPELINE,
        '''        summary = run_gtja185_batch_pipeline(
            output_dir=args.gtja_output,
            start_date=args.gtja_start_date,
''',
        '''        summary = run_gtja185_batch_pipeline(
            output_dir=args.gtja_output,
            dataset=args.gtja_dataset,
            run_mode=args.gtja_run_mode,
            universe_id=args.gtja_universe_id,
            start_date=args.gtja_start_date,
''',
        "pipeline dataset/run mode invocation",
    )
    replace_exact(
        PIPELINE,
        '''            cost_bps=args.gtja_cost_bps,
            universe_dataset=args.gtja_universe_dataset,
''',
        '''            cost_bps=args.gtja_cost_bps,
            fdr_alpha=args.gtja_fdr_alpha,
            universe_dataset=args.gtja_universe_dataset,
''',
        "pipeline FDR invocation",
    )


def patch_docs() -> None:
    text = DOCS.read_text(encoding="utf-8")
    anchor = '''python -m pipeline --gtja185 \\
  --gtja-start-date 2014-01-01 \\
  --gtja-end-date 2026-06-25 \\
  --gtja-horizons 1,5,21 \\
  --gtja-batch-size 8 \\
  --gtja-output /tmp/gtja185_eval
'''
    replacement = '''python -m pipeline --gtja185 \\
  --gtja-run-mode production \\
  --gtja-dataset ashare_stock_daily \\
  --gtja-universe-id A_SHARE_ALL_A_EX_ST \\
  --gtja-require-pit-universe \\
  --gtja-universe-dataset ashare_universe_daily \\
  --gtja-start-date 2014-01-01 \\
  --gtja-end-date 2026-06-25 \\
  --gtja-horizons 1,5,21 \\
  --gtja-entry-lag 1 \\
  --gtja-cost-bps 10 \\
  --gtja-fdr-alpha 0.10 \\
  --gtja-batch-size 8 \\
  --gtja-output /tmp/gtja185_eval
'''
    if anchor not in text:
        raise RuntimeError("GTJA command documentation anchor not found")
    text = text.replace(anchor, replacement, 1)
    text += '''

## Point-in-time universe dataset

`ashare_universe_daily` is parameterized by `universe_id`; each universe is stored under
`universe_daily/universe_id=<id>/date=<date>/`. Required columns are `TradeDate`, `Symbol`,
`is_member`, and `is_tradable`. The evaluator passes the requested universe ID through
DataAccess, so path identity, snapshot lineage and caches cannot mix different universes.
'''
    DOCS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_registry()
    patch_evaluator()
    patch_pipeline()
    patch_docs()
    print("GTJA185 PIT universe dataset parameterized and exposed through CLI")


if __name__ == "__main__":
    main()
