from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.gtja185_batch import (
    BatchEvaluationConfig,
    build_synthetic_market_frame,
    run_gtja185_evaluation,
)
from factor_packs.gtja185 import EXPECTED_FACTOR_COUNT, load_gtja185_pack


def test_gtja185_pack_is_complete_and_versioned():
    pack = load_gtja185_pack()
    assert pack.name == "gtja185"
    assert len(pack.factors) == EXPECTED_FACTOR_COUNT == 185
    assert len(set(pack.names())) == 185
    assert len(pack.pack_hash) == 64
    assert len(pack.source_catalog_hash) == 64
    for factor in pack.factors:
        assert factor.name.startswith("gtja191_alpha_")
        assert factor.formula
        assert len(factor.formula_hash) == 64
        if "VWAP" in factor.source_formula.upper():
            assert "col('vwap')" in factor.formula or 'col("vwap")' in factor.formula


def test_pack_manifests_preserve_factor_identity():
    pack = load_gtja185_pack()
    manifests = [
        factor.to_manifest(start_date="2019-01-01", end_date="2024-12-31")
        for factor in pack.factors
    ]
    assert len(manifests) == 185
    assert {manifest["candidate_id"] for manifest in manifests} == set(pack.names())
    assert all(manifest["factor_pack"] == "gtja185" for manifest in manifests)
    assert all(manifest["expression_type"] == "dsl" for manifest in manifests)


def test_all_185_are_evaluated_on_synthetic_panel(tmp_path: Path):
    frame = build_synthetic_market_frame(periods=420, symbols=8, seed=191)
    config = BatchEvaluationConfig(
        market="ashare",
        horizons=(1, 5),
        batch_size=16,
        min_assets=6,
        n_quantiles=4,
        strict=True,
        resume=False,
    )
    summary = run_gtja185_evaluation(
        config,
        output_dir=tmp_path,
        market_frame=frame,
        snapshot_id="synthetic:gtja185-full-test",
    )
    assert summary.status == "success"
    assert summary.factor_count == 185
    assert summary.succeeded == 185
    assert summary.failed == 0
    ranking = pd.read_csv(tmp_path / "ranking.csv")
    assert len(ranking) == 185
    assert ranking["factor_name"].nunique() == 185
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["pack_hash"] == load_gtja185_pack().pack_hash
    assert payload["snapshot_id"] == "synthetic:gtja185-full-test"
    assert all(record["status"] == "success" for record in payload["records"])
