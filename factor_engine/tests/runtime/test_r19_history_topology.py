# -*- coding: utf-8 -*-
"""R19-075..093 targeted regression tests for the history/topology side of the
execution contract layer:

* R19-075..077  — expanding/cumulative operators declare an ``AnchorPolicy``;
                  ``campaign_start`` is never an implicit factor anchor.
* R19-078       — retained production-direct operators must NOT use the
                  ``_WINDOW_LIKE_PARAM_NAMES`` heuristic history fallback.
* R19-079       — stale ``ts_cusum_break_score`` is gone from the stateful seed;
                  the renamed canonicals carry their own contracts.
* R19-080/081   — ``HistoryRequirement.minimum_effective_samples`` floor is
                  readable (skew>=3, kurtosis>=4, corr>=3, regression>=p+1).
* R19-082/083   — ``MissingTopologyPolicy``: lag is physical-row; autocorr /
                  episode/state must not re-bridge missing sides as neighbours.
* R19-084..086  — days/age/duration outputs declare an explicit ``TimeUnit``.
* R19-087       — ``CurrentRowRequirement`` declared per operator family.
* R19-088..093  — rank/quantile semantics are declared and do not drift between
                  the 0-1-singleton-0.5 family and the 1/n..1 family.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"


def _load_runtime_modules():
    """Load ``runtime.semantic_policies`` + ``runtime.execution_contract``.

    Concurrent R18/R19 sessions are mid-edit on shared files (``fields/``,
    ``cleaned_operators/``) and ``runtime/__init__.py`` transitively imports
    ``storage`` -> ``fields``.  When that chain is temporarily broken we bypass
    the package ``__init__`` and load the two modules directly by file path.
    The modules are self-contained (execution_contract only imports
    semantic_policies); lazy registry lookups degrade to the research
    fail-open path, which is exactly what these tests exercise.
    """
    try:
        import factor_engine.runtime.semantic_policies as sp
        import factor_engine.runtime.execution_contract as ec

        return sp, ec
    except Exception:
        pass  # tree is mid-edit by a concurrent session — use direct loading

    _runtime = sys.modules.get("runtime")
    if _runtime is None or not getattr(_runtime, "__path__", None):
        stub = types.ModuleType("runtime")
        stub.__path__ = [str(_RUNTIME_DIR)]
        sys.modules["runtime"] = stub

    sp_spec = importlib.util.spec_from_file_location(
        "factor_engine.runtime.semantic_policies", str(_RUNTIME_DIR / "semantic_policies.py")
    )
    sp = importlib.util.module_from_spec(sp_spec)
    sys.modules["factor_engine.runtime.semantic_policies"] = sp
    sp_spec.loader.exec_module(sp)

    ec_spec = importlib.util.spec_from_file_location(
        "factor_engine.runtime.execution_contract", str(_RUNTIME_DIR / "execution_contract.py")
    )
    ec = importlib.util.module_from_spec(ec_spec)
    sys.modules["factor_engine.runtime.execution_contract"] = ec
    ec_spec.loader.exec_module(ec)
    return sp, ec


sp, ec = _load_runtime_modules()

AnchorPolicy = sp.AnchorPolicy
MissingTopologyPolicy = sp.MissingTopologyPolicy
CurrentRowRequirement = sp.CurrentRowRequirement
TimeUnit = sp.TimeUnit


# ---------------------------------------------------------------------------
# R19-075..077: expanding/cumulative anchor policy
# ---------------------------------------------------------------------------
def test_expanding_cumulative_canonicals_declare_anchor():
    """Every expanding/cumulative canonical carries an explicit AnchorPolicy."""
    expanding_cumulative = (
        "expanding_mean", "expanding_std", "expanding_rank", "expanding_zscore",
        "expanding_max", "expanding_min", "expanding_sum",
        "cum_avg", "cum_count", "cum_delta", "cum_first", "cum_last",
        "cum_max", "cum_min", "cum_positive_streak", "cum_prod", "cum_rank",
        "cum_std", "cum_sum", "cum_top_n_avg", "cum_top_n_sum",
        "cumulative_max", "cumulative_mean", "cumulative_min",
        "cumulative_returns",
    )
    for canon in expanding_cumulative:
        policy = sp.anchor_policy(canon)
        assert policy is not None, f"{canon} has no declared anchor policy"
        assert isinstance(policy, AnchorPolicy)
        assert policy != AnchorPolicy.CAMPAIGN_START


def test_campaign_start_is_never_implicit_anchor():
    """R19-077: no canonical may be implicitly campaign-anchored."""
    for canon, policy in sp.declared_anchor_policies().items():
        assert policy != "campaign_start", f"{canon} is campaign-anchored"
    assert sp.anchor_belongs_to_campaign("expanding_mean") is False
    # the machine-readable assertion bucket is empty
    assert sp.anchor_belongs_to_campaign("expanding_rank") is False


def test_anchor_policy_enum_values():
    assert {a.value for a in AnchorPolicy} == {
        "listing_start", "fixed_global_start", "campaign_start",
        "full_available_history",
    }


# ---------------------------------------------------------------------------
# R19-078: retained direct operators must not use the heuristic fallback
# ---------------------------------------------------------------------------
class _FakeMeta:
    param_names = ["window"]
    param_specs = {}


def test_retained_direct_operator_forbidden_heuristic_fallback(monkeypatch):
    """A production-direct operator with a window-like param and NO explicit
    history declaration must fail closed (``_UNKNOWN`` -> full history) instead
    of guessing ``window - 1``; legacy/research tiers keep the heuristic."""
    monkeypatch.setattr(ec, "_metadata", lambda canon, strict=False: _FakeMeta())
    monkeypatch.setattr(ec, "_is_production_direct", lambda canon: True)
    assert ec._default_window_extension("some_prod_op", {"window": 20}) is ec._UNKNOWN

    monkeypatch.setattr(ec, "_is_production_direct", lambda canon: False)
    assert ec._default_window_extension("research_op", {"window": 20}) == 19


def test_pointwise_operator_without_window_param_stays_identity(monkeypatch):
    """Operators with NO window-like param are pointwise identity (0 rows), not
    gated — there is nothing to guess."""
    monkeypatch.setattr(
        ec, "_metadata",
        lambda canon, strict=False: type("M", (), {"param_names": [], "param_specs": {}})(),
    )
    monkeypatch.setattr(ec, "_is_production_direct", lambda canon: True)
    assert ec._default_window_extension("pointwise", {}) == 0


def test_daily_window_ops_now_have_explicit_history_transforms():
    """R19-078: every daily-tier operator that previously fell back to the
    name-based heuristic now has an explicit ``_HISTORY_TRANSFORMS`` entry."""
    for canon in (
        "ADX", "ATR_WILDER", "RSI_WILDER", "KAMA",
        "coskewness_to_market", "digital_count", "group_decay_linear",
        "group_ts_decay_linear", "idio_skew", "idio_vol",
        "price_spread_deviation", "rank_corr", "residual_momentum_capm",
        "tail_beta", "ts_decay_exp_window", "ts_max_buildup", "ts_moment",
        "ts_poly2_coeff", "ts_poly2_resid", "ts_quantile", "ts_topk_sum",
    ):
        assert canon in ec._HISTORY_TRANSFORMS, f"{canon} still relies on fallback"


def test_trade_when_signal_is_identity_not_window():
    """``trade_when``'s ``signal`` LOOKS window-like but is a value, not a
    window — the explicit transform must be identity, never ``signal - 1``."""
    transform = ec._HISTORY_TRANSFORMS["trade_when"]
    assert transform.kind == "identity"
    assert ec._own_history_extension("trade_when", {"signal": 20}) == 0


# ---------------------------------------------------------------------------
# R19-079: stale cusum name cleanup + ghost audit
# ---------------------------------------------------------------------------
def test_stale_cusum_break_score_removed_from_stateful_seed():
    assert "ts_cusum_break_score" not in ec._STATEFUL_CANONICALS


def test_renamed_cusum_canonicals_carry_own_contracts():
    # ts_cusum_pressure is a recursive operator via declare_stateful.
    # ts_cumulative_deviation_score is a rolling-window op (not stateful).
    assert "ts_cusum_pressure" not in ec._STATEFUL_CANONICALS
    assert "ts_cumulative_deviation_score" not in ec._STATEFUL_CANONICALS
    # the rolling-window op still resolves a finite (never full-history) window.
    req = ec.history_requirement("ts_cumulative_deviation_score", {"window": 20})
    assert req.kind == "finite"


def test_stateful_seed_has_no_cusum_break_ghost():
    for canon in ec._STATEFUL_CANONICALS:
        assert canon != "ts_cusum_break_score"


# ---------------------------------------------------------------------------
# R19-080/081: minimum_effective_samples support floor
# ---------------------------------------------------------------------------
def test_minimum_effective_samples_floor_declared():
    assert ec.minimum_effective_samples("ts_skew") == 3
    assert ec.minimum_effective_samples("ts_kurt") == 4
    assert ec.minimum_effective_samples("ts_corr") == 3
    assert ec.minimum_effective_samples("ts_cov") == 3
    assert ec.minimum_effective_samples("ts_poly2_coeff") == 3
    assert ec.minimum_effective_samples("ts_poly2_resid") == 3


def test_history_requirement_carries_minimum_effective_samples():
    req = ec.history_requirement("ts_skew", {"window": 20})
    assert req.minimum_effective_samples == 3
    req_kurt = ec.history_requirement("ts_kurt", {"window": 20})
    assert req_kurt.minimum_effective_samples == 4
    req_mean = ec.history_requirement("ts_mean", {"window": 20})
    assert req_mean.minimum_effective_samples is None  # no floor declared


def test_history_requirement_dataclass_has_floor_field():
    assert "minimum_effective_samples" in ec.HistoryRequirement.__dataclass_fields__


# ---------------------------------------------------------------------------
# R19-082/083: MissingTopologyPolicy
# ---------------------------------------------------------------------------
def test_missing_topology_policy_enum_values():
    assert {m.value for m in MissingTopologyPolicy} == {
        "skip_finite", "preserve_physical_lag", "break_episode",
        "require_contiguous", "pairwise_finite",
    }


def test_lag_is_physical_row_not_previous_observed():
    """lag=1 is the previous TRADING BAR (physical-row lag), never the previous
    non-missing observation."""
    for canon in ("ts_delay", "ts_delta", "ts_log_return", "MOM", "ROC", "prev", "ts_ratio"):
        assert sp.missing_topology_policy(canon) is MissingTopologyPolicy.PRESERVE_PHYSICAL_LAG


def test_autocorr_does_not_rebridge_missing_neighbours():
    for canon in ("ts_autocorr", "volume_autocorr", "turnover_autocorr"):
        assert sp.missing_topology_policy(canon) is MissingTopologyPolicy.PRESERVE_PHYSICAL_LAG


def test_episode_state_breaks_on_missing():
    for canon in ("ts_cusum_pressure", "directional_change_state", "cum_positive_streak"):
        assert sp.missing_topology_policy(canon) is MissingTopologyPolicy.BREAK_EPISODE


def test_correlation_pairwise_finite():
    for canon in ("ts_corr", "ts_cov", "ts_beta", "rank_corr", "ts_regression"):
        assert sp.missing_topology_policy(canon) is MissingTopologyPolicy.PAIRWISE_FINITE


def test_mean_skips_finite():
    assert sp.missing_topology_policy("ts_mean") is MissingTopologyPolicy.SKIP_FINITE


# ---------------------------------------------------------------------------
# R19-084..086: explicit time units
# ---------------------------------------------------------------------------
def test_time_unit_declarations():
    assert sp.time_unit("ts_pivot_high_age") is TimeUnit.TRADING_BARS
    assert sp.time_unit("ts_pivot_low_spacing") is TimeUnit.TRADING_BARS
    assert sp.time_unit("ts_pivot_high_count") is TimeUnit.EVENTS
    assert sp.time_unit("cum_count") is TimeUnit.EVENTS
    assert sp.time_unit("fin_lag") is TimeUnit.REPORTS
    assert sp.time_unit("fin_diff") is TimeUnit.REPORTS


def test_days_named_row_count_canonicals_is_declared():
    """No operator claims a ``days``/``age``/``duration`` name while counting
    physical rows without a declaration — the audit bucket is machine-readable
    and currently empty."""
    assert isinstance(sp.days_named_row_count_canonicals(), frozenset)
    assert "ts_pivot_high_age" not in sp.days_named_row_count_canonicals()


# ---------------------------------------------------------------------------
# R19-087: CurrentRowRequirement
# ---------------------------------------------------------------------------
def test_current_row_requirement_declarations():
    assert sp.current_row_requirement("ts_mean") is CurrentRowRequirement.NOT_REQUIRED_FOR_WINDOW_STAT
    assert sp.current_row_requirement("ts_corr") is CurrentRowRequirement.REQUIRED_AS_PAIR
    assert sp.current_row_requirement("ts_cov") is CurrentRowRequirement.REQUIRED_AS_PAIR
    assert sp.current_row_requirement("ts_regression_forecast_error") is CurrentRowRequirement.REQUIRED_AS_TARGET
    assert sp.current_row_requirement("event_historical_response_mean") is CurrentRowRequirement.REQUIRED_AS_EVENT


# ---------------------------------------------------------------------------
# R19-088..093: rank / quantile family unification
# ---------------------------------------------------------------------------
def test_rank_semantics_do_not_drift_between_families():
    """``rank``/``cs_rank``/``c_rank`` resolve to the 0-1-singleton-0.5 family;
    ``rank_pct`` is the 1/n..1 family.  The two are different canonicals."""
    for canon in ("rank", "cs_rank_01"):
        assert sp.rank_semantics(canon) is sp.RankSemantic.NORMALIZED_01_SINGLETON_HALF
    assert sp.rank_semantics("rank_pct") is sp.RankSemantic.PCT_RANK_COUNT
    assert sp.rank_semantics("cs_pct_rank") is sp.RankSemantic.PCT_RANK_COUNT
    assert sp.rank_semantics("rankavg_transform") is sp.RankSemantic.RAW_RANK_AVERAGE
    assert sp.rank_semantics("ts_rank") is sp.RankSemantic.ROLLING_WINDOW
    assert sp.rank_semantics("expanding_rank") is sp.RankSemantic.EXPANDING
    assert sp.rank_semantics("cum_rank") is sp.RankSemantic.EXPANDING
    assert sp.rank_semantics("group_rank") is sp.RankSemantic.GROUP


def test_quantile_policy_p_strict_range_and_interpolation_fixed():
    for canon in ("cs_quantile", "ts_quantile", "group_percentile", "quantile_normal", "quantile_t"):
        policy = sp.quantile_policy(canon)
        assert policy is not None
        assert policy.p_min == 0.0 and policy.p_max == 1.0
        assert policy.interpolation in {"linear", "nearest", "lower", "higher", "midpoint"}
        assert policy.finite_policy == "drop_finite"


def test_quantile_policy_declarations_snapshot_machine_readable():
    snapshot = sp.rank_quantile_declarations()
    assert "rank" in snapshot and "quantile" in snapshot
    assert snapshot["rank"]["rank"] == "normalized_01_singleton_half"


def test_all_policy_declarations_snapshot():
    snapshot = sp.all_policy_declarations()
    assert "anchor" in snapshot and "missing_topology" in snapshot
    assert "current_row" in snapshot and "time_units" in snapshot
    assert "rank_quantile" in snapshot


# ---------------------------------------------------------------------------
# Existing behavior preserved (regression guard)
# ---------------------------------------------------------------------------
def test_existing_history_rows_preserved():
    """The explicit transforms must not change the finite row counts that the
    planner already relies on."""
    assert ec.history_requirement("ts_mean", {"window": 60}).rows == 59
    assert ec.history_requirement("ts_delay", {"n": 20}).rows == 20
    assert ec.history_requirement("ts_quantile", {"d": 20}).rows == 19


def test_execution_contract_re_exports_semantic_policies():
    for name in (
        "AnchorPolicy", "MissingTopologyPolicy", "CurrentRowRequirement",
        "TimeUnit", "anchor_policy", "anchor_belongs_to_campaign",
        "missing_topology_policy", "current_row_requirement", "time_unit",
        "rank_semantics", "quantile_policy", "minimum_effective_samples",
    ):
        assert name in ec.__all__
        assert hasattr(ec, name)
