# -*- coding: utf-8 -*-
"""R24-019..022: holder ratio units come ONLY from the declared source
contract.  No data-value heuristic (``max(ratio)>1 → /100``); out-of-contract
values raise a source-contract error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.storage.sources.relation import aggregate_holder_rows


def _rows(ratios):
    return pd.DataFrame({
        "instrument": ["A"] * len(ratios),
        "period_end": pd.to_datetime(["2024-03-31"] * len(ratios)),
        "available_at": pd.to_datetime(["2024-04-30"] * len(ratios)),
        "holding_amount": list(range(len(ratios))),
        "holding_ratio": ratios,
    })


def test_percent_contract_scales_to_decimal() -> None:
    # R24-021: raw percent [0.3, 0.8] → 0.3% / 0.8%, NOT 30% / 80%.
    agg = aggregate_holder_rows(_rows([0.3, 0.8]), ratio_unit="percent")
    assert np.isclose(agg.loc[0, "top_ten_holding_ratio"], 0.003 + 0.008)


def test_decimal_contract_identity() -> None:
    agg = aggregate_holder_rows(_rows([0.03, 0.08]), ratio_unit="decimal")
    assert np.isclose(agg.loc[0, "top_ten_holding_ratio"], 0.03 + 0.08)


def test_out_of_contract_value_raises_source_contract_error() -> None:
    # R24-022: [0.008, 2.0] under a decimal contract is a source-contract
    # violation — never a silent rescale.
    with pytest.raises(ValueError, match="source-contract error"):
        aggregate_holder_rows(_rows([0.008, 2.0]), ratio_unit="decimal")


def test_no_unit_raises_no_guessing() -> None:
    with pytest.raises(ValueError, match="ratio_unit"):
        aggregate_holder_rows(_rows([0.05, 0.06]))


def test_negative_ratio_outside_contract() -> None:
    with pytest.raises(ValueError, match="source-contract error"):
        aggregate_holder_rows(_rows([-0.1, 0.05]), ratio_unit="decimal")
