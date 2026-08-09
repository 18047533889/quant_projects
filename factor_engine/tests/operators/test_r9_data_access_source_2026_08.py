"""Round-9 review items for storage/sources/data_access_source.py.

Covers:
  R9-P0-022  strictness must come from run_mode/production, never self-inferred
             from the global environment.
  R9-P0-023  covers_window() and violations() share one window-aware evaluator.
  R9-P0-024  coverage_by_stock (and per-date coverage) actually gate.

These tests exercise the pure Python gate methods and the strictness resolution
directly (no dataset file IO, no catalog load) so they stay hermetic.
"""
from __future__ import annotations

import pytest


def _contract(**kwargs):
    from storage.sources.data_access_source import HistoricalCoverageContract

    base = dict(field="r9_test_field")
    base.update(kwargs)
    return HistoricalCoverageContract(**base)


# ---------------------------------------------------------------------------
# R9-P0-023 — shared window-aware evaluator
# ---------------------------------------------------------------------------
def test_covers_window_ignores_out_of_window_years():
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2015: 0.1, 2024: 0.9},
        threshold=0.7,
    )
    # 2024 is covered; 2015 (outside the window) must not veto it.
    assert c.covers_window(start="2024-01-01", end="2024-12-31")
    # The same requested window must be respected by violations() too — this is
    # the exact bug where violations() iterated ALL coverage_by_year.
    assert c.violations(start="2024-01-01", end="2024-12-31") == []


def test_covers_window_and_violations_share_evaluator():
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2015: 0.1, 2024: 0.9},
        threshold=0.7,
    )
    # Identical inputs must produce identical verdicts.
    assert c.covers_window(start="2024-01-01", end="2024-12-31") == (
        not c.violations(start="2024-01-01", end="2024-12-31")
    )
    # Without a window the full history IS the window: 2015 low coverage flags.
    assert not c.covers_window()
    assert any("year 2015" in p for p in c.violations())


def test_assert_historical_coverage_is_window_aware():
    from storage.sources.data_access_source import assert_historical_coverage

    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2015: 0.1, 2024: 0.9},
        threshold=0.7,
    )
    # A requested window must NOT be rejected because of out-of-window low coverage.
    assert_historical_coverage(c, start="2024-01-01", end="2024-12-31")
    # Without a window the low 2015 year is inside the (full-history) window.
    with pytest.raises(Exception):
        assert_historical_coverage(c)


def test_in_window_year_still_gates():
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2023: 0.3, 2024: 0.9},
        threshold=0.7,
    )
    assert not c.covers_window(start="2023-01-01", end="2023-12-31")
    assert any("year 2023" in p for p in c.violations(start="2023-01-01", end="2023-12-31"))
    assert c.covers_window(start="2024-01-01", end="2024-12-31")


# ---------------------------------------------------------------------------
# R9-P0-024 — coverage_by_stock and per-date coverage actually gate
# ---------------------------------------------------------------------------
def test_coverage_by_stock_gate_flags_tail_stock():
    # High aggregate (0.9) and high per-year coverage, but one stock at 5%.
    by_stock = {f"s{i}": 0.95 for i in range(4)}
    by_stock["s_bad"] = 0.05
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_stock=by_stock,
        threshold=0.7,
    )
    problems = c.violations(start="2024-01-01", end="2024-12-31")
    assert any("stock" in p for p in problems), problems
    assert not c.covers_window(start="2024-01-01", end="2024-12-31")
    # A stock at 5% coverage is a real cross-sectional hole: message names it.
    assert any("4/5" in p or "80.0%" in p for p in problems), problems


def test_coverage_by_stock_clean_data_passes():
    by_stock = {f"s{i}": 0.95 for i in range(5)}
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_stock=by_stock,
        threshold=0.7,
    )
    assert c.violations(start="2024-01-01", end="2024-12-31") == []
    assert c.covers_window(start="2024-01-01", end="2024-12-31")


def test_coverage_by_stock_gate_is_quantile_tunable():
    by_stock = {f"s{i}": 0.95 for i in range(4)}
    by_stock["s_bad"] = 0.05
    # With a 100% quantile requirement the single bad stock still fails…
    strict = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_stock=by_stock,
        threshold=0.7,
        min_stock_coverage_quantile=1.0,
    )
    assert not strict.covers_window(start="2024-01-01", end="2024-12-31")
    # …and lowering the quantile below the observed fraction passes it.
    loose = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_stock=by_stock,
        threshold=0.7,
        min_stock_coverage_quantile=0.7,
    )
    assert loose.covers_window(start="2024-01-01", end="2024-12-31")


def test_per_date_coverage_gates_within_window():
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_date={"2024-01-02": 0.95, "2024-01-03": 0.3},
        threshold=0.7,
    )
    problems = c.violations(start="2024-01-01", end="2024-12-31")
    assert any("2024-01-03" in p and "date" in p for p in problems), problems
    assert not c.covers_window(start="2024-01-01", end="2024-12-31")
    # A date outside the requested window is not evaluated.
    assert c.violations(start="2024-01-02", end="2024-01-02") == []


def test_clean_per_date_data_passes():
    c = _contract(
        coverage_ratio=0.9,
        coverage_by_year={2024: 0.9},
        coverage_by_date={"2024-01-02": 0.95, "2024-01-03": 0.97},
        threshold=0.7,
    )
    assert c.violations(start="2024-01-01", end="2024-12-31") == []


# ---------------------------------------------------------------------------
# R9-P0-022 — strictness from run_mode/production, not the global env
# ---------------------------------------------------------------------------
def _source_with_gate_only(**kwargs):
    """Build a DataAccessSource that exercises only the strict-gate branch.

    The catalog / store / preflight paths are short-circuited so the test checks
    the run-mode strictness resolution without touching dataset IO.
    """
    from storage.sources.data_access_source import DataAccessSource

    src = DataAccessSource(dataset="ashare_stock_daily", **kwargs)
    src._preflight_logical_columns = lambda names: None
    src._resolve_catalog_fields = lambda names: {}
    src._field_spec = lambda name: None
    src._semantic_catalog_version = lambda: "test-token"
    return src


def test_run_mode_production_rejects_unknown_field():
    from storage.sources.data_access_source import UnknownFieldSemanticError

    src = _source_with_gate_only(run_mode="production")
    assert src.strict_unknown_fields is True
    assert src.production is True
    assert src.run_mode == "production"
    with pytest.raises(UnknownFieldSemanticError):
        src._resolve_columns(["no_such_column"])


def test_run_mode_research_allows_unknown_field():
    src = _source_with_gate_only(run_mode="research")
    assert src.strict_unknown_fields is False
    assert src.production is False
    physical, _ = src._resolve_columns(["no_such_column"])
    assert physical == ["no_such_column"]


def test_production_flag_and_strict_unknown_fields_backward_compatible():
    from storage.sources.data_access_source import UnknownFieldSemanticError

    # production=True behaves like strict_unknown_fields=True.
    src = _source_with_gate_only(production=True)
    assert src.strict_unknown_fields is True
    with pytest.raises(UnknownFieldSemanticError):
        src._resolve_columns(["no_such_column"])

    # Explicit strict_unknown_fields still wins (existing callers keep working).
    src = _source_with_gate_only(strict_unknown_fields=True, run_mode="research")
    assert src.strict_unknown_fields is True
    with pytest.raises(UnknownFieldSemanticError):
        src._resolve_columns(["no_such_column"])

    # production=False wins over run_mode="production".
    src = _source_with_gate_only(production=False, run_mode="production")
    assert src.strict_unknown_fields is False
    physical, _ = src._resolve_columns(["no_such_column"])
    assert physical == ["no_such_column"]


def test_default_run_mode_is_research_never_self_inferred(monkeypatch):
    # Even if the environment claims production, an unqualified constructor must
    # NOT self-infer strictness — it fails open to research.
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    src = _source_with_gate_only()
    assert src.strict_unknown_fields is False
    assert src.production is False
    assert src.run_mode is None
    physical, _ = src._resolve_columns(["no_such_column"])
    assert physical == ["no_such_column"]
