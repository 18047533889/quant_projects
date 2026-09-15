from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

NAMES = ("ts_first_passage_hit_probability", "ts_first_passage_conditional_time")


def _op(name, backend="pandas_numpy"):
    op = OperatorRegistry.get(name, backend=backend)
    assert op is not None
    return op


def _oracle(x, scale, *, window, barrier, horizon, min_anchors, side, conditional):
    result = np.full(len(x), np.nan)
    for t in range(len(x)):
        outcomes = []
        for s in range(max(0, t - window), t - horizon + 1):
            if not np.isfinite(x[s]) or not np.isfinite(scale[s]) or scale[s] <= 0:
                continue
            path = x[s + 1:s + horizon + 1]
            if len(path) != horizon or not np.all(np.isfinite(path)):
                continue
            direction, tau = 0, 0
            for k, value in enumerate(path, 1):
                if value >= x[s] + barrier * scale[s]:
                    direction, tau = 1, k
                    break
                if value <= x[s] - barrier * scale[s]:
                    direction, tau = -1, k
                    break
            outcomes.append((direction, tau))
        if len(outcomes) < min_anchors:
            continue
        target = 1 if side == "upper" else -1
        hits = [tau for direction, tau in outcomes if direction == target]
        if conditional:
            if len(hits) >= min_anchors:
                result[t] = np.mean(hits) / horizon
        else:
            result[t] = len(hits) / len(outcomes)
    return result


def _call(name, x, scale, backend="pandas_numpy", **kwargs):
    index = pd.date_range("2026-01-01", periods=len(x))
    xp = pd.DataFrame({"B": x, "A": np.asarray(x) * .7}, index=index)
    sp = pd.DataFrame({"B": scale, "A": np.asarray(scale) * .7}, index=index)
    if backend == "polars":
        xp, sp = pl.from_pandas(xp.reset_index(drop=True)), pl.from_pandas(sp.reset_index(drop=True))
    out = _op(name, backend).calculate(xp, sp, **kwargs)
    return out.to_pandas() if isinstance(out, pl.DataFrame) else out


def test_fresh_load_exact_contract_twins_and_defaults():
    expected = ["x", "scale", "window", "barrier", "horizon", "min_anchors", "scale_horizon", "side"]
    for name in NAMES:
        pd_op, pl_op = _op(name), _op(name, "polars")
        assert pd_op.metadata == pl_op.metadata
        meta = pd_op.metadata
        assert meta.param_names == expected
        assert meta.panel_params == ("x", "scale") and meta.panel_arity == 2
        assert meta.scalar_params == tuple(expected[2:])
        assert set(meta.param_specs) == set(expected[2:])
        assert {key: spec.default for key, spec in meta.param_specs.items()} == {
            "window": 120, "barrier": 1.0, "horizon": 10,
            "min_anchors": 3, "scale_horizon": 1, "side": "upper",
        }
        assert meta.param_specs["window"].history_semantics == "max_rows"


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_matches_independent_stopping_time_oracle(backend):
    x = np.array([0., 2., 2.2, 1.9, 4., 4.1, 3.9, 7., 7.2, 6.8])
    scale = np.ones_like(x)
    kwargs = dict(window=7, barrier=1., horizon=2, min_anchors=2, scale_horizon=1)
    for name in NAMES:
        for side in ("upper", "lower"):
            actual = _call(name, x, scale, backend, side=side, **kwargs).iloc[:, 0].to_numpy()
            expected = _oracle(x, scale, side=side, conditional=name.endswith("conditional_time"), **{k: kwargs[k] for k in ("window", "barrier", "horizon", "min_anchors")})
            np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=0, atol=1e-15)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_affine_scale_invariance_missing_paths_and_prefix_causality(backend):
    x = np.array([0., .2, 1.5, 1.7, .1, -.2, -1.5, -1.7, 0., 1.4, 1.6, .2])
    scale = np.full_like(x, .8)
    kwargs = dict(window=8, barrier=1., horizon=2, min_anchors=2, side="upper")
    for name in NAMES:
        base = _call(name, x, scale, backend, **kwargs)
        transformed = _call(name, x * 1e120 + 3e120, scale * 1e120, backend, **kwargs)
        pd.testing.assert_frame_equal(base, transformed)
        longer = _call(name, np.r_[x, 1e200], np.r_[scale, .8], backend, **kwargs)
        pd.testing.assert_frame_equal(base.reset_index(drop=True), longer.iloc[:-1].reset_index(drop=True))
        missing = x.copy()
        missing[3] = np.nan
        got = _call(name, missing, scale, backend, **kwargs)
        expected_missing = _oracle(
            missing, scale, conditional=name.endswith("conditional_time"),
            **kwargs,
        )
        np.testing.assert_allclose(got.iloc[:, 0], expected_missing, equal_nan=True)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_domain_rejection_and_backend_default_parity(backend):
    x = pd.DataFrame({"A": np.arange(20.0)})
    scale = pd.DataFrame({"A": np.ones(20)})
    if backend == "polars":
        x, scale = pl.from_pandas(x), pl.from_pandas(scale)
    for name in NAMES:
        op = _op(name, backend)
        for bad_kwargs in (
            {"side": "both"}, {"barrier": 0.0}, {"barrier": np.inf},
            {"window": 5, "horizon": 6},
            {"window": 8, "horizon": 3, "min_anchors": 7},
            {"scale_horizon": 1.5},
        ):
            with pytest.raises((TypeError, ValueError)):
                op.calculate(x, scale, **bad_kwargs)
    if backend == "polars":
        pdf_x, pdf_s = x.to_pandas(), scale.to_pandas()
        for name in NAMES:
            expected = _op(name).calculate(pdf_x, pdf_s)
            actual = _op(name, "polars").calculate(x, scale).to_pandas()
            pd.testing.assert_frame_equal(expected, actual)
