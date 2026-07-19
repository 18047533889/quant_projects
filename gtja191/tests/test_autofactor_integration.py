from __future__ import annotations

import importlib.util
import unittest

_HAS_EVAL = importlib.util.find_spec("evaluation") is not None


@unittest.skipUnless(_HAS_EVAL, "AutoFactorEvaluation evaluation package not installed")
class TestAutoFactorIntegration(unittest.TestCase):
    """Full AFV evaluator tests; skipped when evaluation.* is absent."""

    def test_gtja_pack_is_external_complete_and_versioned(self) -> None:
        from autofactor.provider import load_pack

        pack = load_pack()
        self.assertEqual(pack.name, "gtja185")
        self.assertEqual(len(pack.factors), 185)
        self.assertEqual(len(set(pack.names())), 185)
        self.assertEqual(len(pack.pack_hash), 64)
        self.assertEqual(len(pack.source_hash), 64)
        self.assertEqual(pack.metadata.get("dsl_surface"), "compat")
        for factor in pack.factors:
            if "VWAP" in factor.source_formula.upper():
                self.assertTrue(
                    "col('vwap')" in factor.formula or 'col("vwap")' in factor.formula
                )


# Keep pytest-style tests for environments that have evaluation + pytest.
if _HAS_EVAL:
    import json
    from pathlib import Path

    import pandas as pd

    from autofactor.provider import load_pack
    from evaluation.batch import (
        BatchEvaluationConfig,
        build_synthetic_market_frame,
        run_factor_pack_evaluation,
    )

    def test_gtja_pack_is_external_complete_and_versioned():
        pack = load_pack()
        assert pack.name == "gtja185"
        assert len(pack.factors) == 185
        assert len(set(pack.names())) == 185
        assert len(pack.pack_hash) == 64
        assert len(pack.source_hash) == 64
        assert pack.metadata.get("dsl_surface") == "compat"
        for factor in pack.factors:
            if "VWAP" in factor.source_formula.upper():
                assert "col('vwap')" in factor.formula or 'col("vwap")' in factor.formula

    def test_all_185_run_through_generic_evaluator(tmp_path: Path):
        pack = load_pack()
        frame = build_synthetic_market_frame(periods=320, symbols=8, seed=191)
        config = BatchEvaluationConfig(
            market="ashare",
            horizons=(1, 5),
            batch_size=16,
            min_assets=6,
            n_quantiles=4,
            factor_version=f"{pack.name}.{pack.version}",
            strict=True,
            resume=False,
        )
        summary = run_factor_pack_evaluation(
            pack,
            config,
            output_dir=tmp_path,
            market_frame=frame,
            snapshot_id="synthetic:gtja185-external",
        )
        assert summary.status == "success"
        assert summary.factor_count == 185
        assert summary.succeeded == 185
        assert summary.failed == 0
        assert sum(summary.route_counts.values()) == 185

        ranking = pd.read_csv(tmp_path / "ranking.csv")
        assert len(ranking) == 185
        assert ranking["factor_name"].nunique() == 185
        assert {
            "validation_rank_ic",
            "validation_rank_ic_ir",
            "validation_rank_ic_q_value",
            "validation_long_short_net_hac_sharpe",
            "test_rank_ic",
            "test_long_short_net_hac_sharpe",
        } <= set(ranking.columns)

        payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert payload["pack_hash"] == pack.pack_hash
        assert payload["source_hash"] == pack.source_hash
