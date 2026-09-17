import numpy as np
import pandas as pd
import pytest

from factor_engine.api.dsl_parser import DSLParser, DSLUnknownOperatorError
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.catalog_r20_statistical_adapters import LEGACY_TO_REVIEWED
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_reviewed_statistical_aliases_have_research_runtime_and_daily_stays_closed():
    load_all()
    research = DSLParser(surface="compat_research")
    daily = DSLParser(surface="daily")
    for old, canonical in LEGACY_TO_REVIEWED.items():
        assert OperatorRegistry.resolve_canonical(old) == canonical
        assert OperatorRegistry.get(canonical, mode="any") is not None
        argc = len(OperatorRegistry.get(canonical, mode="any").metadata.panel_params)
        args = ", ".join(["ret", "turnover_ratio"][:argc])
        assert research.parse(f"{old}({args})").op == canonical
        with pytest.raises(DSLUnknownOperatorError):
            daily.parse(f"{old}({args})")


def test_jarque_bera_adapter_is_prefix_causal_and_panel_shaped():
    load_all()
    op = OperatorRegistry.get("ts_expanding_jarque_bera_pvalue", mode="any")
    idx = pd.date_range("2025-01-01", periods=40)
    panel = pd.DataFrame({"A": np.linspace(-0.03, 0.04, 40), "B": np.sin(np.arange(40))}, index=idx)
    full = op.calculate(panel)
    prefix = op.calculate(panel.iloc[:25])
    assert full.shape == panel.shape
    pd.testing.assert_frame_equal(full.iloc[:25], prefix)
