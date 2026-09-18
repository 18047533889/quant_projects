"""Regression contracts for the reviewed expanding Lilliefors p-value."""
import numpy as np
import pandas as pd
import pytest
import re
import tomllib
from pathlib import Path

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _operator():
    load_all()
    return OperatorRegistry.get("ts_expanding_lilliefors_pvalue", "pandas_numpy", mode="any")


def test_expanding_lilliefors_known_reference_values():
    frame = pd.DataFrame({"A": np.arange(6.0)})
    actual = _operator().calculate(frame)["A"].to_numpy()
    assert np.isnan(actual[:4]).all()
    np.testing.assert_allclose(actual[4], 0.9890685554219967, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual[5], 0.99, rtol=0, atol=1e-15)


def test_expanding_lilliefors_uses_only_finite_prefix_support():
    values = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, np.inf, 5.0])
    actual = _operator().calculate(pd.DataFrame({"A": values}))["A"].to_numpy()
    assert np.isnan(actual[:5]).all()
    np.testing.assert_allclose(actual[5], 0.9890685554219967, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual[6], actual[5], rtol=0, atol=0)
    np.testing.assert_allclose(actual[7], 0.99, rtol=0, atol=1e-15)


def test_expanding_lilliefors_prefix_is_future_invariant_and_degenerate_is_nan():
    rng = np.random.default_rng(4401)
    values = rng.normal(size=30)
    op = _operator()
    before = op.calculate(pd.DataFrame({"A": values}))["A"].to_numpy()
    changed = values.copy()
    changed[20:] = changed[20:] * 50 + 100
    after = op.calculate(pd.DataFrame({"A": changed}))["A"].to_numpy()
    np.testing.assert_allclose(before[:20], after[:20], equal_nan=True)
    constant = op.calculate(pd.DataFrame({"A": np.ones(8)}))["A"].to_numpy()
    assert np.isnan(constant).all()


def test_missing_statsmodels_is_explicit_and_does_not_block_other_statistics(monkeypatch):
    import builtins
    from factor_engine.cleaned_operators.common import statistics

    real_import = builtins.__import__

    def reject_statsmodels(name, *args, **kwargs):
        if name == "statsmodels.stats.diagnostic":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_statsmodels)
    frame = pd.DataFrame({"A": np.arange(6.0)})
    with pytest.raises(ImportError, match="requires statsmodels"):
        _operator().calculate(frame)
    actual = statistics.mean_agg().calculate(frame)
    np.testing.assert_allclose(actual["A"].to_numpy(), np.arange(6.0) / 2.0)


def test_factor_engine_declares_lilliefors_dependency_and_team_lock_pins_it():
    factor_engine_root = Path(__file__).resolve().parents[2]
    repository_root = factor_engine_root.parent
    metadata = tomllib.loads((factor_engine_root / "pyproject.toml").read_text())
    dependencies = metadata["project"]["dependencies"]
    assert "statsmodels>=0.14" in dependencies
    production_lock = (repository_root / "requirements-production.lock").read_text()
    assert re.search(r"^statsmodels==0\.14\.6\s+--hash=sha256:[0-9a-f]{64}$", production_lock, re.MULTILINE)
