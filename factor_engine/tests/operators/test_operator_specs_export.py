# -*- coding: utf-8
"""OperatorSpec manifest 导出与新增算子测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy operator export inventory superseded by generated manifest convergence")

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_spec import (
    PRODUCTION_CORE_CANONICALS,
    export_operator_manifest,
    spec_to_manifest_entry,
    build_operator_spec,
)
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_ts_log_return(_loaded):
    op = OperatorRegistry.get("ts_log_return")
    x = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    out = op.calculate(x, 1)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(np.log(1.1))
    assert out.iloc[2, 0] == pytest.approx(np.log(121 / 110))


def test_operating_margin_zero_revenue(_loaded):
    op = OperatorRegistry.get("operating_margin")
    income = pd.DataFrame({"A": [10.0, 20.0]})
    revenue = pd.DataFrame({"A": [0.0, 100.0]})
    out = op.calculate(income, revenue)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(0.2)


def test_current_ratio_zero_liabilities(_loaded):
    op = OperatorRegistry.get("current_ratio")
    assets = pd.DataFrame({"A": [200.0, 300.0]})
    liabilities = pd.DataFrame({"A": [0.0, 100.0]})
    out = op.calculate(assets, liabilities)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(3.0)


def test_quick_ratio_excludes_inventory(_loaded):
    op = OperatorRegistry.get("quick_ratio")
    assets = pd.DataFrame({"A": [200.0]})
    inventory = pd.DataFrame({"A": [50.0]})
    liabilities = pd.DataFrame({"A": [100.0]})
    out = op.calculate(assets, inventory, liabilities)
    assert out.iloc[0, 0] == pytest.approx(1.5)


def test_debt_to_equity_zero_equity(_loaded):
    op = OperatorRegistry.get("debt_to_equity")
    debt = pd.DataFrame({"A": [100.0, 200.0]})
    equity = pd.DataFrame({"A": [0.0, 50.0]})
    out = op.calculate(debt, equity)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(4.0)


def test_ts_sharpe_and_autocorr_have_polars_backend(_loaded):
    assert "polars" in OperatorRegistry.backends_for("ts_sharpe")
    assert "polars" in OperatorRegistry.backends_for("ts_autocorr")


def test_ts_std_single_observation_is_nan(_loaded):
    op = OperatorRegistry.get("ts_std")
    x = pd.DataFrame({"A": [1.0]})
    out = op.calculate(x, 5)
    assert pd.isna(out.iloc[0, 0])


def test_spec_to_manifest_entry_core(_loaded):
    spec = build_operator_spec("ts_mean")
    assert spec is not None
    entry = spec_to_manifest_entry(spec)
    assert entry["name"] == "ts_mean"
    assert entry["pit_safe"] is True
    assert entry["scope"] == "time_series"
    assert "x" in entry["input_fields"] or entry["input_fields"]


def test_export_operator_manifest_core_subset(_loaded):
    entries = export_operator_manifest(core_only=True)
    names = {e["name"] for e in entries}
    assert "ts_mean" in names
    assert "rolling_beta" in names
    assert "ts_sharpe" in names
    assert "ts_autocorr" in names
    assert len(names) == len(PRODUCTION_CORE_CANONICALS)


def test_export_script_writes_json(_loaded, tmp_path):
    from scripts.export_operator_specs import main

    out = tmp_path / "specs.json"
    assert main(["-o", str(out), "--core-only"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert any(e["name"] == "zscore" for e in data)
