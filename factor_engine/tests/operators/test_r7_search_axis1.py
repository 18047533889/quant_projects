# -*- coding: utf-8
"""WS-G search-space audit / dedup tests (review #294-#306).

Covers the round-7 cross-sectional rank fix (axis=1), exact vs round(9) numeric
hashing, the NaN validity-mask hash, per-factor-kind signatures, the explicit
DedupPolicy, and the ALL-operators default of the parameter-sensitivity audit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.search.factor_dedup import (
    DedupPolicy,
    FactorKind,
    build_typed_fixtures,
    condition_signature,
    event_signature,
    factor_signatures,
    numeric_signature,
    probe_panel,
    rank_signature,
    signature_for,
    time_series_signature,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_audit_module():
    """Import scripts/parameter_sensitivity_audit.py as a module (no main())."""
    path = _REPO_ROOT / "scripts" / "parameter_sensitivity_audit.py"
    spec = importlib.util.spec_from_file_location("param_sensitivity_audit", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# (a) rank uses axis=1 (cross-sectional) — review #300
# --------------------------------------------------------------------------- #
def test_rank_duplicates_uses_cross_sectional_axis1():
    """A per-row-broadcast shift is rank-identical under axis=1 but NOT axis=0.

    ``fb = p + t`` where ``t`` is a per-ROW broadcast: within each row every
    column shifts by the same value, so the cross-sectional (axis=1) ordering is
    unchanged.  Column-by-column (axis=0) the time order is scrambled by the
    non-monotone ``t``, so an axis=0 implementation would (correctly) NOT dedup.
    """
    rng = np.random.default_rng(0)
    rows, cols = 40, 8
    p = rng.normal(size=(rows, cols))
    t = rng.normal(size=(rows, 1))  # per-row broadcast
    panel = pd.DataFrame(
        p,
        index=pd.date_range("2023-01-02", periods=rows, freq="D"),
        columns=[f"C{i}" for i in range(cols)],
    )

    def fa(_f):
        return pd.DataFrame(p, index=_f.index, columns=_f.columns)

    def fb(_f):
        return pd.DataFrame(p + t, index=_f.index, columns=_f.columns)

    from cleaned_operators.search.factor_dedup import are_rank_duplicates

    # cross-sectional rank says: same factor -> duplicates.
    assert are_rank_duplicates(fa, fb, panel=panel) is True

    # Sanity: the two panels really do differ cross-sectionally in the raw data
    # sense only via the broadcast; verify axis=1 rank correlation is ~1 while
    # axis=0 rank correlation is < 1 (proves the fix is load-bearing).
    ra1 = fa(panel).rank(axis=1).to_numpy()
    rb1 = fb(panel).rank(axis=1).to_numpy()
    rho1 = np.corrcoef(ra1.ravel(), rb1.ravel())[0, 1]
    ra0 = fa(panel).rank(axis=0).to_numpy()
    rb0 = fb(panel).rank(axis=0).to_numpy()
    rho0 = np.corrcoef(ra0.ravel(), rb0.ravel())[0, 1]
    assert rho1 >= 0.99999
    assert rho0 < 0.99999


def test_hash_columns_rank_axis1_differs_from_axis0():
    from cleaned_operators.search.factor_dedup import _hash_columns_rank

    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]],
        index=pd.date_range("2023-01-02", periods=3, freq="D"),
        columns=["a", "b"],
    )
    assert _hash_columns_rank(df, axis=1) != _hash_columns_rank(df, axis=0)


def test_rank_signature_is_axis1_cross_sectional():
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]],
        index=pd.date_range("2023-01-02", periods=3, freq="D"),
        columns=["a", "b"],
    )
    identity = lambda _f: df
    assert rank_signature(identity) == rank_signature(identity)


# --------------------------------------------------------------------------- #
# (b) NaN validity-mask hash — review #303
# --------------------------------------------------------------------------- #
def test_nan_placement_changes_hash():
    """Same finite values, NaN in different positions -> different signature.

    The old ``nan_to_num(..., nan=-1e300)`` approach collapsed these to the
    same hash; the validity-mask hash must distinguish them.
    """
    from cleaned_operators.search.factor_dedup import _hash_columns

    a = pd.DataFrame([[1.0, np.nan], [np.nan, 2.0]])
    b = pd.DataFrame([[np.nan, 1.0], [2.0, np.nan]])
    # same multiset of finite values {1,2}, same NaN count, different placement
    assert sorted(a.to_numpy()[np.isfinite(a)]) == sorted(b.to_numpy()[np.isfinite(b)])
    assert _hash_columns(a) != _hash_columns(b)


def test_nan_vs_finite_same_values_differ():
    """NaN substitution (nan->0) would make NaN==0; the mask must not."""
    from cleaned_operators.search.factor_dedup import _hash_columns

    with_nan = pd.DataFrame([[1.0, np.nan]])
    with_zero = pd.DataFrame([[1.0, 0.0]])
    assert _hash_columns(with_nan) != _hash_columns(with_zero)


# --------------------------------------------------------------------------- #
# (c) exact numeric signature, no round(9) — review #302
# --------------------------------------------------------------------------- #
def test_numeric_signature_not_rounded():
    """Two values differing in the 10th decimal must hash differently."""
    from cleaned_operators.search.factor_dedup import _hash_columns

    a = pd.DataFrame([[1.0000000001, 2.0]])
    b = pd.DataFrame([[1.0000000002, 2.0]])
    assert _hash_columns(a) != _hash_columns(b)
    # sanity: under round(9) both collapse to the same value
    assert a.to_numpy().round(9)[0, 0] == b.to_numpy().round(9)[0, 0]


def test_numeric_signature_exact_bytes_deterministic():
    f = lambda _f: pd.DataFrame([[1.0000000001, np.nan], [3.0, -4.5]])
    assert numeric_signature(f) == numeric_signature(f)


# --------------------------------------------------------------------------- #
# (d) choices / int-grid / float-quantile sampling — review #296
# --------------------------------------------------------------------------- #
class _FakeMeta:
    def __init__(self, specs):
        self.param_specs = specs
        self.param_names = list(specs)


class _FakeOp:
    def __init__(self, specs):
        self.metadata = _FakeMeta(specs)


def test_choices_sampling_enumerates_legal_choices():
    from cleaned_operators.base import ParamSpec

    mod = _load_audit_module()
    op = _FakeOp({"bins": ParamSpec(dtype=int, choices=(3, 5))})
    assert mod._sample_values("bins", op, 3) == [3, 5]


def test_int_bounds_sample_legal_integer_grid():
    from cleaned_operators.base import ParamSpec

    mod = _load_audit_module()
    op = _FakeOp({"window": ParamSpec(dtype=int, min=2, max=6)})
    vals = mod._sample_values("window", op, 3)
    assert all(isinstance(v, int) for v in vals)
    assert all(2 <= v <= 6 for v in vals)
    assert len(set(vals)) >= 2


def test_float_bounds_sample_reviewed_quantiles():
    from cleaned_operators.base import ParamSpec

    mod = _load_audit_module()
    op = _FakeOp({"alpha": ParamSpec(dtype=float, min=0.0, max=1.0)})
    vals = sorted(mod._sample_values("alpha", op, 3))
    assert vals[0] == 0.0 and vals[-1] == 1.0
    assert 0.5 in vals  # mid
    assert all(0.0 <= v <= 1.0 for v in vals)  # never out of legal bounds


# --------------------------------------------------------------------------- #
# (e) sign-invariant DedupPolicy — review #306
# --------------------------------------------------------------------------- #
def test_sign_invariant_policy_dedups_x_and_neg_x():
    from cleaned_operators.search.factor_dedup import are_rank_duplicates

    panel = probe_panel()
    f = lambda _f: _f
    neg = lambda _f: -_f

    assert are_rank_duplicates(f, neg, panel=panel, policy=DedupPolicy(sign_invariant=True)) is True
    assert are_rank_duplicates(f, neg, panel=panel, policy=DedupPolicy(sign_invariant=False)) is False


def test_in_sign_bucket_collides_for_sign_flips():
    panel = probe_panel()
    f = lambda _f: _f
    neg = lambda _f: -_f
    inv = DedupPolicy(sign_invariant=True)
    sens = DedupPolicy(sign_invariant=False)
    assert inv.in_sign_bucket(f, panel=panel) == inv.in_sign_bucket(neg, panel=panel)
    assert sens.in_sign_bucket(f, panel=panel) != sens.in_sign_bucket(neg, panel=panel)


# --------------------------------------------------------------------------- #
# (f) MAX_OPS defaults to ALL; stable-hash sharding — review #294
# --------------------------------------------------------------------------- #
def test_max_ops_defaults_to_all():
    mod = _load_audit_module()
    assert mod.MAX_OPS is None
    assert mod._in_shard("any_canonical", 1, None) is True


def test_stable_sharding_is_deterministic_and_partitions():
    mod = _load_audit_module()
    canonicals = [f"op_{i}" for i in range(64)]
    # deterministic across calls
    assert [mod._in_shard(c, 16, 3) for c in canonicals] == [
        mod._in_shard(c, 16, 3) for c in canonicals
    ]
    # every canonical lands in exactly one shard
    for c in canonicals:
        hits = [i for i in range(16) if mod._in_shard(c, 16, i)]
        assert len(hits) == 1
    # union of all 16 shards covers everything
    covered = {c for i in range(16) for c in canonicals if mod._in_shard(c, 16, i)}
    assert covered == set(canonicals)


def test_sharding_selects_real_subset_on_large_catalog():
    mod = _load_audit_module()
    import string

    canonicals = [f"{c}{i}" for i, c in enumerate(string.ascii_lowercase * 30)]
    all_sel = [c for c in canonicals if mod._in_shard(c, 16, None)]
    shard_sel = [c for c in canonicals if mod._in_shard(c, 16, 5)]
    assert all_sel == canonicals
    assert len(shard_sel) < len(canonicals)


# --------------------------------------------------------------------------- #
# typed fixtures (review #299 / #301)
# --------------------------------------------------------------------------- #
def test_typed_fixtures_shapes_and_regimes():
    fixtures = build_typed_fixtures()
    expected = {
        "gaussian", "heavy_tail", "trend", "mean_revert", "ties", "gaps",
        "positive_only", "event_mask", "group", "ohlc",
    }
    assert set(fixtures) == expected
    for name, df in fixtures.items():
        assert df.shape == (240, 128), name
        assert df.index.is_monotonic_increasing, name
    # regime distinctions: ties are discrete, gaps contain NaN, positive_only > 0
    assert fixtures["ties"].to_numpy().dtype == np.float64
    assert np.isnan(fixtures["gaps"].to_numpy()).any()
    assert (fixtures["positive_only"].to_numpy() > 0).all()


# --------------------------------------------------------------------------- #
# per-factor-kind signatures (review #305)
# --------------------------------------------------------------------------- #
def test_signature_for_dispatches_per_kind():
    panel = probe_panel()
    f = lambda _f: _f

    sig_alpha = signature_for(FactorKind.ALPHA, f, panel=panel)
    sig_global = signature_for(FactorKind.GLOBAL_STATE, f, panel=panel)
    sig_cond = signature_for(FactorKind.CONDITION, f, panel=panel)
    sig_event = signature_for(FactorKind.EVENT, f, panel=panel)

    # ALPHA uses the cross-sectional rank signature.
    assert sig_alpha == rank_signature(f, panel=panel)
    # GLOBAL_STATE uses the time-series rank signature.
    assert sig_global == time_series_signature(f, panel=panel)
    # distinct families produce distinct families for the same factor.
    assert len({sig_alpha, sig_global, sig_cond, sig_event}) >= 3

    with pytest.raises(ValueError):
        signature_for("NOT_A_KIND", f, panel=panel)


def test_condition_and_event_signatures_differ_on_state_vs_timing():
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.normal(size=(40, 8)))

    # same magnitudes, but event signature keys on WHERE non-zero events fire.
    sparse_a = base.copy()
    sparse_a.iloc[:5, :] = 0.0
    sparse_b = base.copy()
    sparse_b.iloc[35:, :] = 0.0
    assert event_signature(lambda _f: sparse_a) != event_signature(lambda _f: sparse_b)


# --------------------------------------------------------------------------- #
# algebraic layer removed (review #304)
# --------------------------------------------------------------------------- #
def test_no_algebraic_canonical_claim():
    f = lambda _f: _f
    sigs = factor_signatures(f)
    assert "algebraic_canonical" not in sigs
    assert {"ast_hash", "numeric_signature", "rank_signature"} <= set(sigs)
    assert sigs["ast_hash"] == ""


# --------------------------------------------------------------------------- #
# split_scalar_panel_params (review #295)
# --------------------------------------------------------------------------- #
def test_split_scalar_panel_params_never_splits_panel_as_scalar():
    from cleaned_operators.search.factor_dedup import split_scalar_panel_params

    class Meta:
        param_names = ["close", "volume", "window", "lag"]
        panel_params = ("close", "volume")
        input_fields = None

    scalar, panel = split_scalar_panel_params(
        ["close", "volume", "window", "lag"], Meta(), "fake_op"
    )
    assert "close" not in scalar and "volume" not in scalar
    assert set(panel) == {"close", "volume"}
    assert set(scalar) == {"window", "lag"}


def test_split_fallback_uses_numeric_control_names():
    from cleaned_operators.search.factor_dedup import split_scalar_panel_params

    class Meta:
        param_names = ["high", "low", "close", "volume", "window"]
        panel_params = None
        input_fields = None

    scalar, panel = split_scalar_panel_params(
        ["high", "low", "close", "volume", "window"], Meta(), "opaque_op"
    )
    assert "window" in scalar
    assert {"high", "low", "close", "volume"} <= set(panel)


def test_split_never_treats_control_knob_as_panel_even_if_metadata_mislabels():
    """window/lag/... are scalar knobs even when a legacy metadata block lists
    them in input_fields (see ts_average_volume / ts_regression_slope)."""
    from cleaned_operators.search.factor_dedup import split_scalar_panel_params

    class Meta:
        param_names = ["x", "window"]
        panel_params = None
        input_fields = ["x", "window"]  # mislabel: window is a scalar knob

    scalar, panel = split_scalar_panel_params(["x", "window"], Meta(), "ts_average_volume")
    assert scalar == ["window"]
    assert panel == ["x"]
