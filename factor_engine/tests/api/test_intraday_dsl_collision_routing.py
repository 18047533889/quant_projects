import numpy as np
import pandas as pd
import pytest

from factor_engine.api.dsl_parser import DSLParseError, DSLParser
from factor_engine.api.intraday_daily import (
    INTRADAY_DAILY_DSL_FUNCTIONS,
    augment_intraday_daily_allowlist,
)
from factor_engine.api.source_ref import decode_source_ref
from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_all_25_collisions_preserve_zero_input_and_explicit_panel_routes():
    cleaned = {}
    for surface in ("daily", "extended", "research"):
        cleaned.update(build_cleaned_dsl_allowlist(set(), surface=surface))
    collisions = sorted(set(cleaned).intersection(INTRADAY_DAILY_DSL_FUNCTIONS))
    assert len(collisions) == 25

    markers = {}
    panel_allow = {}
    for name in collisions:
        marker = object()
        markers[name] = marker
        panel_allow[name] = lambda *args, _marker=marker, **kwargs: _marker
    routed = augment_intraday_daily_allowlist(panel_allow)

    for name in collisions:
        source_expr = routed[name]()
        assert decode_source_ref(source_expr.name) is not None, name
        assert routed[name](object()) is markers[name], name


def _minute_inputs():
    stamps = []
    for offset in range(2):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=offset)
        stamps.extend(
            day + pd.Timedelta(minutes=m)
            for m in (*range(571, 691), *range(781, 901))
        )
    index = pd.DatetimeIndex(stamps)
    returns = 0.0015 * np.sin(np.arange(len(index)) / 7.0) + 0.0003
    close = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(returns))}, index=index)
    amount = close * (10.0 + 2e4 * np.abs(returns[:, None]))
    return close, amount


def test_legacy_panel_dsl_keeps_drawdown_jump_and_amihud_arguments():
    parser = DSLParser(surface="compat_research")
    close, amount = _minute_inputs()
    cases = (
        ("intra_max_drawdown(minute_close)", "intra_max_drawdown", (close,), {}),
        ("intra_jump_count(minute_close, 2.5)", "intra_jump_count", (close, 2.5), {"threshold_scale": 2.5}),
        ("intra_amihud(minute_close, minute_amount, 1e8)", "intra_amihud", (close, amount, 1e8), {"scale": 1e8}),
    )
    for formula, canonical, positional, keyword in cases:
        expr = parser.parse(formula)
        assert expr.op == canonical
        assert len(expr.args) == len(positional)
        operator = OperatorRegistry.get(canonical, mode="any")
        by_position = operator.calculate(*positional)
        if keyword:
            by_keyword = operator.calculate(*positional[: len(positional) - 1], **keyword)
            np.testing.assert_allclose(by_position, by_keyword, equal_nan=True)
        assert np.isfinite(by_position.to_numpy(dtype=float)).any()


def test_zero_input_collision_still_builds_source_ref():
    parser = DSLParser(surface="compat_research")
    for name in ("intra_max_drawdown", "intra_jump_count", "intra_amihud"):
        spec = decode_source_ref(parser.parse(f"{name}()").name)
        assert spec is not None
        assert spec.transform == "intraday_feature"


def test_named_panel_input_routes_to_cleaned_and_mixed_source_controls_reject():
    parser = DSLParser(surface="compat_research")
    assert parser.parse("intra_max_drawdown(close=minute_close)").op == "intra_max_drawdown"
    with pytest.raises(DSLParseError, match="cannot mix explicit panel inputs"):
        parser.parse("intra_max_drawdown(minute_close, bar_minutes=5)")
