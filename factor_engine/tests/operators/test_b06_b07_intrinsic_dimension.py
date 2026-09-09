from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.intrinsic_dimension as intrinsic
from factor_engine.cleaned_operators.base import RelationalParamSpec
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_relational_ast_none_conditional_is_lazy_and_still_restricted() -> None:
    relation = RelationalParamSpec(
        "n_embed >= k + (dim * delay if theiler is None else theiler) + 1"
    )
    assert relation.param_names == {"n_embed", "k", "dim", "delay", "theiler"}
    assert relation.check({"n_embed": 9, "k": 5, "dim": 3, "delay": 1, "theiler": None})
    assert relation.check({"n_embed": 6, "k": 5, "dim": 3, "delay": 1, "theiler": 0})
    assert not relation.check({"n_embed": 8, "k": 5, "dim": 3, "delay": 1, "theiler": None})

    lazy = RelationalParamSpec("7 if optional is None else 1 / 0")
    assert lazy.check({"optional": None})
    assert not lazy.check({"optional": 1})
    with pytest.raises(ValueError, match="identity comparison only with None"):
        RelationalParamSpec("left is right")
    with pytest.raises(ValueError, match="unsupported node Call"):
        RelationalParamSpec("__import__('os')")


def _panel(seed: int = 7, rows: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(size=rows).cumsum()})


def test_theiler_impossible_domain_rejects_but_zero_and_boundary_are_feasible() -> None:
    op = OperatorRegistry.get("ts_delay_intrinsic_dimension", "pandas_numpy", mode="any")
    x = _panel()
    with pytest.raises(ValueError, match=r"N_embed >= k \+ effective_theiler \+ 1"):
        op.calculate(x, window=8, embedding_dim=3, delay=1, k=5)

    zero = op.calculate(
        x, window=8, embedding_dim=3, delay=1, k=5, theiler_window=0
    )
    assert np.isfinite(zero["A"]).any()

    boundary = op.calculate(x, window=11, embedding_dim=3, delay=1, k=5)
    assert np.isfinite(boundary["A"]).any()
    explicit_boundary = op.calculate(
        x, window=11, embedding_dim=3, delay=1, k=5, theiler_window=3
    )
    np.testing.assert_allclose(boundary, explicit_boundary, equal_nan=True)

    with pytest.raises(ValueError, match=r"N_embed >= k \+ effective_theiler \+ 1"):
        op.calculate(
            x, window=11, embedding_dim=3, delay=1, k=5, theiler_window=4
        )


def test_intrinsic_distance_is_stable_under_extreme_global_scaling_and_translation() -> None:
    rng = np.random.default_rng(20260909)
    values = rng.normal(size=20).cumsum()
    reference = intrinsic._delay_intrinsic_dim(values, 3, 3, 1, 0)
    assert np.isfinite(reference)
    for transformed in (
        values * 1e-200,
        values * 1e200,
        values + 1e8,
    ):
        actual = intrinsic._delay_intrinsic_dim(transformed, 3, 3, 1, 0)
        assert actual == pytest.approx(reference, rel=2e-5, abs=1e-10)

    # Preserve the high-offset stress case. Adding 1e12 quantizes the original
    # increments at roughly 1e-4 before the operator receives them, so compare
    # against the unshifted estimate with a commensurate output tolerance.
    high_offset = intrinsic._delay_intrinsic_dim(values + 1e12, 3, 3, 1, 0)
    assert high_offset == pytest.approx(reference, rel=1e-4, abs=1e-10)


def test_duplicate_cloud_is_undefined_not_zero_or_infinite() -> None:
    actual = intrinsic._delay_intrinsic_dim(np.ones(20), 3, 3, 1, 0)
    assert np.isnan(actual)
