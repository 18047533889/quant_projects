"""Small mock-only serialization contracts for real optimizer diagnostics."""
from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
          "benchmark_raw_summary_cache_oct04.py")
SPEC = importlib.util.spec_from_file_location(
    "benchmark_raw_summary_cache_oct04_diagnostic_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_real_shaped_diagnostics_and_config_are_json_serializable():
    from factor_optimizer.research_batch import BatchOptimizationConfig

    # Representative nested structures emitted by diagnose_training_batch,
    # assess_baseline_training, and the optimizer's candidate ledger.
    training = {
        "selection_portfolio": {
            "status": "available",
            "metrics": {"sharpe": np.float64(0.75),
                        "worst_block_sharpe": np.float64(0.2)},
            "periods_per_year": np.int64(252),
        },
        "issues": [{"code": "unstable_ic_direction",
                    "families": [],
                    "value": [np.float64(-0.1), np.float64(0.2)]}],
        "layer_decay": {
            "status": "available",
            "layers": [{"layer": np.int64(1),
                        "mean_excess_returns": np.asarray([0.01, -0.02]),
                        "stable_initial_direction": np.bool_(True),
                        "half_life_bars": None}],
            "proposed_half_lives": [np.float64(5.0)],
        },
        "rank_ic": np.float64(0.03),
        "fold_rank_ic": [np.float64(0.01), None, np.float64(0.05)],
    }
    baseline = {
        "operations": ("winsor", "cs_rank"),
        "omissions": (),
        "accepted": np.bool_(False),
        "reason": "training-only gate",
        "raw_metrics": {"coverage": np.float64(0.95),
                        "folds": np.asarray([0.1, 0.2, 0.3])},
    }
    config = asdict(BatchOptimizationConfig())

    encoded = BENCH._canonical_json(BENCH._jsonable({
        "config": config,
        "training_diagnostics": training,
        "baseline_diagnostics": baseline,
    }))

    assert '"training_diagnostics"' in encoded
    assert '"baseline_diagnostics"' in encoded
    assert '"operations":["winsor","cs_rank"]' in encoded
    assert '"families":[]' in encoded


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_diagnostic_scalar_fails_closed_before_report_encoding(bad):
    normalized = BENCH._jsonable({"training_diagnostics": {"metric": np.float64(bad)}})

    with pytest.raises(ValueError, match="Out of range float values"):
        BENCH._canonical_json(normalized)
