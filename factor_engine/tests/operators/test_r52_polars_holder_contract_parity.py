"""Deep parity coverage for contract-repaired common Polars holder operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()


_PARAMS = {
    "holder_concentration": 2,
    "holder_concentration_acceleration": 3,
    "holder_concentration_slope": 3,
    "holder_topk_share_sum": 20,
    "holder_company_ownership_hhi": 10,
    "holder_observed_topk_hhi": 11,
    "holder_class_entropy": 8,
    "holder_nature_entropy": 8,
    "holder_disclosure_count": 20,
    "holder_disclosure_coverage": 20,
    "holder_entry_share": 40,
    "holder_exit_share": 40,
    "holder_net_entry_share": 40,
    "holder_id_matched_entry_share": 40,
    "holder_id_matched_exit_share": 40,
    "holder_id_matched_churn": 40,
    "holder_weighted_churn": 40,
    "holder_rank_stability": 40,
    "holder_id_overlap_ratio": 40,
    "holder_shareholder_overlap_ratio": 2,
    "holder_float_concentration_gap": 2,
    "holder_freeze_concentration": 10,
    "holder_pledge_concentration": 10,
    "holder_pledge_churn": 20,
    "holder_pledged_holder_count": 10,
    "holder_common_holding_peer_return": 3,
    "holder_share_weighted_rank_migration": 40,
    "holder_shareholder_network_centrality": 2,
    "holder_class_js_shift": 10,
}


def _panel(value: float, *, nan_last: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=6, freq="D")
    data = np.full((6, 2), value, dtype=float)
    data[:, 1] *= 1.5
    if nan_last:
        data[-1, 1] = np.nan
    return pd.DataFrame(data, index=index, columns=["A", "B"])


def _to_polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.rename_axis("date").reset_index())


def _to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    out = frame.to_pandas()
    return out.set_index("date") if "date" in out else out


def _id_args() -> list[pd.DataFrame]:
    current_shares = [_panel(0.10 if i < 2 else 0.0) for i in range(10)]
    current_ids = [_panel(float(i + 1)) for i in range(10)]
    previous_shares = [_panel(0.08 if i in (0, 2) else 0.0) for i in range(10)]
    previous_ids = [_panel(float(i + 1 if i != 1 else 20)) for i in range(10)]
    return current_shares + current_ids + previous_shares + previous_ids


def _args(canonical: str):
    shares10 = [_panel(0.02 * (i + 1), nan_last=(i == 9)) for i in range(10)]
    disclosure20 = shares10 + [_panel(float(i + 1)) for i in range(10)]
    ranked20 = shares10 + [_panel(0.01 * (i + 1)) for i in range(10)]
    if canonical == "holder_concentration":
        return [_panel(20.0, nan_last=True), _panel(100.0)]
    if canonical in {"holder_concentration_slope", "holder_concentration_acceleration"}:
        trend = _panel(0.0)
        trend.loc[:, "A"] = np.arange(6, dtype=float)
        trend.loc[:, "B"] = np.arange(6, dtype=float) ** 2
        return [trend, 3, None]
    if canonical in {"holder_topk_share_sum", "holder_disclosure_count", "holder_disclosure_coverage"}:
        return disclosure20
    if canonical in {"holder_company_ownership_hhi", "holder_freeze_concentration", "holder_pledge_concentration", "holder_pledged_holder_count"}:
        return shares10
    if canonical == "holder_observed_topk_hhi":
        return [*shares10, "outside_top_k"]
    if canonical in {"holder_class_entropy", "holder_nature_entropy"}:
        return shares10[:8]
    if canonical in {
        "holder_entry_share", "holder_exit_share", "holder_net_entry_share",
        "holder_id_matched_entry_share", "holder_id_matched_exit_share",
        "holder_id_matched_churn", "holder_weighted_churn", "holder_rank_stability",
        "holder_id_overlap_ratio", "holder_share_weighted_rank_migration",
    }:
        return _id_args()
    if canonical == "holder_pledge_churn":
        return ranked20
    if canonical == "holder_common_holding_peer_return":
        return [_panel(0.03, nan_last=True), _panel(0.01), _panel(0.2)]
    if canonical == "holder_shareholder_overlap_ratio":
        return [_panel(2.0, nan_last=True), _panel(5.0)]
    if canonical == "holder_float_concentration_gap":
        return [_panel(0.4, nan_last=True), _panel(0.1)]
    if canonical == "holder_shareholder_network_centrality":
        return [_panel(2.0, nan_last=True), _panel(10.0)]
    if canonical == "holder_class_js_shift":
        return shares10
    raise AssertionError(canonical)


@pytest.mark.parametrize("canonical", sorted(_PARAMS))
def test_repaired_holder_public_contract_and_reference_parity(canonical):
    reference = OperatorRegistry.get(canonical, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(canonical, backend="polars")
    assert reference is not None and polars_op is not None
    assert len(reference.metadata.param_names) == _PARAMS[canonical]
    assert polars_op.metadata.param_names == reference.metadata.param_names

    args = _args(canonical)
    polars_args = [_to_polars(arg) if isinstance(arg, pd.DataFrame) else arg for arg in args]
    expected = reference.calculate(*args)
    actual = _to_pandas(polars_op.calculate(*polars_args))

    assert actual.index.equals(expected.index), canonical
    assert actual.columns.equals(expected.columns), canonical
    assert np.isfinite(expected.to_numpy(dtype=float)).any(), canonical
    assert np.isfinite(actual.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float), expected.to_numpy(dtype=float),
        equal_nan=True, atol=1e-10, err_msg=canonical,
    )
