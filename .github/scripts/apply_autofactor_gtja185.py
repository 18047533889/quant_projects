from __future__ import annotations

import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"


def generate_pack() -> None:
    runpy.run_path(str(PROJECT / "scripts" / "generate_gtja185_pack.py"), run_name="__main__")


def add_gateway_wrappers() -> None:
    gateway = PROJECT / "gateway"
    wrappers = {
        "config": "gateway.scripts.config",
        "complexity": "gateway.scripts.complexity",
        "deduplicator": "gateway.scripts.deduplicator",
        "future_scanner": "gateway.scripts.future_scanner",
        "io_utils": "gateway.scripts.io_utils",
        "kafka_producer": "gateway.scripts.kafka_producer",
        "validator": "gateway.scripts.validator",
        "router": "gateway.scripts.router",
        "data_quality": "gateway.scripts.data_quality",
    }
    for name, module in wrappers.items():
        (gateway / f"{name}.py").write_text(
            f'"""Compatibility import for {module}."""\nfrom {module} import *  # noqa: F401,F403\n',
            encoding="utf-8",
        )
    (gateway / "gateway_core.py").write_text(
        '"""Compatibility façade for both Gateway APIs."""\n'
        'from gateway.gateway.gateway_core import *  # noqa: F401,F403\n'
        'from gateway.scripts.gateway_core import run_gateway  # noqa: F401\n',
        encoding="utf-8",
    )


def remove_fastparquet_runtime_dependency() -> None:
    for path in PROJECT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        updated = re.sub(r",\s*engine=[\"']fastparquet[\"']", "", text)
        updated = re.sub(r"engine=[\"']fastparquet[\"']\s*,\s*", "", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def patch_pipeline_cli() -> None:
    path = PROJECT / "pipeline.py"
    text = path.read_text(encoding="utf-8")
    if "def run_gtja185_batch_pipeline(" not in text:
        anchor = "# ============================================================\n# CLI\n# ============================================================\n"
        addition = '''def run_gtja185_batch_pipeline(\n    *,\n    output_dir: str | Path,\n    start_date: str | None = None,\n    end_date: str | None = None,\n    horizons: tuple[int, ...] = (1, 5, 21),\n    batch_size: int = 8,\n    min_assets: int = 20,\n    limit: int | None = None,\n    synthetic: bool = False,\n    materialize_staging: bool = False,\n    publish: bool = False,\n    strict: bool = True,\n):\n    """Run the canonical 185-factor pack through the new batch evaluator."""\n    from evaluation.gtja185_batch import (\n        BatchEvaluationConfig,\n        build_synthetic_market_frame,\n        run_gtja185_evaluation,\n    )\n\n    config = BatchEvaluationConfig(\n        market="ashare",\n        start_date=start_date,\n        end_date=end_date,\n        horizons=horizons,\n        batch_size=batch_size,\n        min_assets=min_assets,\n        limit=limit,\n        materialize_staging=materialize_staging,\n        publish=publish,\n        strict=strict,\n    )\n    frame = build_synthetic_market_frame(periods=420, symbols=max(8, min_assets)) if synthetic else None\n    snapshot_id = "synthetic:gtja185-cli" if synthetic else None\n    return run_gtja185_evaluation(\n        config,\n        output_dir=output_dir,\n        market_frame=frame,\n        snapshot_id=snapshot_id,\n    )\n\n\n'''
        if anchor not in text:
            raise RuntimeError("pipeline CLI anchor not found")
        text = text.replace(anchor, addition + anchor, 1)

    parser_anchor = '    p.add_argument("--gateway-only", action="store_true", help="仅 Gateway 审查+路由")\n'
    if "--gtja185" not in text:
        parser_args = '''    p.add_argument("--gtja185", action="store_true", help="评估完整 GTJA185 内置因子包")\n    p.add_argument("--gtja-output", default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185")\n    p.add_argument("--gtja-start-date", default=None)\n    p.add_argument("--gtja-end-date", default=None)\n    p.add_argument("--gtja-horizons", default="1,5,21")\n    p.add_argument("--gtja-batch-size", type=int, default=8)\n    p.add_argument("--gtja-min-assets", type=int, default=20)\n    p.add_argument("--gtja-limit", type=int, default=None)\n    p.add_argument("--gtja-synthetic", action="store_true")\n    p.add_argument("--gtja-materialize-staging", action="store_true")\n    p.add_argument("--gtja-publish", action="store_true")\n    p.add_argument("--gtja-allow-partial", action="store_true")\n'''
        if parser_anchor not in text:
            raise RuntimeError("pipeline parser anchor not found")
        text = text.replace(parser_anchor, parser_args + parser_anchor, 1)

    execute_anchor = "    args = p.parse_args()\n\n"
    if "if args.gtja185:" not in text:
        execute = '''    args = p.parse_args()\n\n    if args.gtja185:\n        from dataclasses import asdict\n\n        horizons = tuple(int(part.strip()) for part in args.gtja_horizons.split(",") if part.strip())\n        summary = run_gtja185_batch_pipeline(\n            output_dir=args.gtja_output,\n            start_date=args.gtja_start_date,\n            end_date=args.gtja_end_date,\n            horizons=horizons,\n            batch_size=args.gtja_batch_size,\n            min_assets=args.gtja_min_assets,\n            limit=args.gtja_limit,\n            synthetic=args.gtja_synthetic,\n            materialize_staging=args.gtja_materialize_staging,\n            publish=args.gtja_publish,\n            strict=not args.gtja_allow_partial,\n        )\n        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str))\n        return\n\n'''
        if execute_anchor not in text:
            raise RuntimeError("pipeline parse_args anchor not found")
        text = text.replace(execute_anchor, execute, 1)
    path.write_text(text, encoding="utf-8")


def patch_assetization_worker_bug() -> None:
    path = PROJECT / "assetization" / "scripts" / "worker.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("mdp, len(market_df), len(market_df.columns)", "bar_dir, len(market_df), len(market_df.columns)")
    path.write_text(text, encoding="utf-8")


def add_docs() -> None:
    path = PROJECT / "docs" / "gtja185_batch_evaluation.md"
    path.write_text(
        """# GTJA185 batch evaluation\n\n"
        "AutoFactorEvaluation now ships a versioned bundle of all 185 deliverable GTJA191 factors. "
        "The bundle is generated from `gtja191/lib/catalog.py` and CI rejects stale copies.\n\n"
        "The official path is `evaluation.gtja185_batch`: one DataAccess snapshot, current FactorEngine, "
        "PIT compilation, cross-sectional MAD winsorization/z-scoring, train-only direction selection, "
        "validation/test IC, RankIC, ICIR, quantile long-short, turnover, annualized performance, drawdown, "
        "year stability, routing, resumable reports, and optional factor_lake_staging writes.\n\n"
        "```bash\n"
        "python -m pipeline --gtja185 --gtja-start-date 2014-01-01 --gtja-end-date 2026-06-25 "
        "--gtja-output /tmp/gtja185_eval\n"
        "```\n\n"
        "For a data-free deterministic smoke run, append `--gtja-synthetic`. Published factor values are "
        "never written directly: use `--gtja-materialize-staging`; add `--gtja-publish` only after review.\n"
        """,
        encoding="utf-8",
    )


def main() -> None:
    generate_pack()
    add_gateway_wrappers()
    remove_fastparquet_runtime_dependency()
    patch_pipeline_cli()
    patch_assetization_worker_bug()
    add_docs()
    print("GTJA185 AutoFactorEvaluation migration applied")


if __name__ == "__main__":
    main()
