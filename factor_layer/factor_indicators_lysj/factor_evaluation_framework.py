"""Backward-compatible CLI entry point for the factor evaluation framework.

This script reads local parquet files and drives the modular evaluation
package.  For programmatic use, import ``evaluation`` directly::

    from evaluation import EvalConfig, run_evaluation, serialize_results
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation import EvalConfig, run_evaluation, serialize_results, print_brief


def main() -> None:
    base_dir = Path(__file__).parent

    # ── Load inputs ───────────────────────────────────────────────────
    factor_df = pd.read_parquet(base_dir / "factor_output.parquet")
    market_df = pd.read_parquet(base_dir / "NQ.parquet")

    cfg = EvalConfig(
        timestamp_col="data",
        vwap_col="average",
        factor_col="factor",
        factor_timestamp_col="timestamp",
        horizons=(1, 5, 10, 20),
    )

    # ── Run pipeline ──────────────────────────────────────────────────
    raw_results = run_evaluation(factor_df, market_df, cfg)

    # ── Console summary ───────────────────────────────────────────────
    print_brief(raw_results)

    # ── Serialise & save JSON ─────────────────────────────────────────
    json_output = serialize_results(raw_results, cfg)
    out_path = base_dir / "evaluation_output" / "evaluation_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
