# -*- coding: utf-8 -*-
"""R40 missing/NaN + stats items #187/#188/#192/#194/#195/#196/#197."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators._rolling_fast import (
    WMA_PARTIAL_POLICY,
    check_wma_partial_policy,
    wma_partial_policy_digest,
)
from cleaned_operators.common.cross_sectional import _finite_stats_input, check_rank_method
from cleaned_operators.common.data_cleaning import _nan_only_to_num, _nonfinite_to_num
from cleaned_operators.common.time_series import (
    EWMContract,
    StatisticalSamplePolicy,
    TopKContract,
    TopKSamplePolicy,
    TopKTiePolicy,
    check_topk_contract,
    ewm_contract_for,
    uniform_finite_mask,
)


class TestNanToNumSemanticClarity:
    def test_nan_to_num_only_replaces_nan(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        out = _nan_only_to_num(arr, num=0.0)
        # NaN -> 0; ±Inf passes through unchanged
        assert out[0] == 1.0
        assert out[1] == 0.0
        assert np.isposinf(out[2])
        assert np.isneginf(out[3])

    def test_nonfinite_to_num_maps_inf_too(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        out = _nonfinite_to_num(arr, num=0.0)
        assert list(out) == [1.0, 0.0, 0.0, 0.0]


class TestFiniteSamplePolicy:
    def test_uniform_finite_mask(self):
        df = pd.DataFrame([[1.0, np.inf], [np.nan, 2.0]])
        mask = uniform_finite_mask(df)
        assert bool(mask.iloc[0, 0]) is True
        assert bool(mask.iloc[0, 1]) is False  # Inf is missing
        assert bool(mask.iloc[1, 0]) is False  # NaN is missing
        assert bool(mask.iloc[1, 1]) is True

    def test_stats_fallback_rejects_inf(self):
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        finite = _finite_stats_input(arr)
        assert np.isnan(finite[0, 1])  # Inf -> NaN
        # np.nanmean over the finite-masked row ignores the Inf
        assert np.nanmean(_finite_stats_input(np.array([[1.0, np.inf]]))) == 1.0

    def test_statistical_sample_policy(self):
        p = StatisticalSamplePolicy(valid="finite_only", min_count=2)
        assert p.min_count == 2
        df = pd.DataFrame([[1.0, np.inf]])
        masked = p.masked(df)
        assert np.isnan(masked.iloc[0, 1])


class TestTopKContract:
    def test_undeclared_tie_policy_rejected_in_production(self):
        with pytest.raises(ValueError):
            check_topk_contract(TopKContract(sample_policy=TopKSamplePolicy.FINITE_ONLY), canonical="topk")

    def test_declared_tie_policy_passes(self):
        check_topk_contract(
            TopKContract(sample_policy=TopKSamplePolicy.FINITE_ONLY, tie_policy=TopKTiePolicy.STABLE_INSTRUMENT_KEY),
            canonical="topk",
        )

    def test_unknown_sample_policy_rejected(self):
        with pytest.raises(ValueError):
            check_topk_contract(TopKContract(sample_policy="bogus", tie_policy=TopKTiePolicy.STABLE_INSTRUMENT_KEY))


class TestEWMContract:
    def test_ema_declares_contract(self):
        contract = ewm_contract_for("ts_ema")
        assert contract is not None
        assert contract.decay_mapping == "span"
        assert contract.adjust is False
        assert contract.digest() != ""

    def test_ewm_contract_digest_enters_identity(self):
        a = EWMContract(canonical="ts_ema", decay_mapping="span", adjust=False)
        b = EWMContract(canonical="ts_ema", decay_mapping="span", adjust=True)
        assert a.digest() != b.digest()


class TestWMAPolicy:
    def test_policy_digest_enters_identity(self):
        d = wma_partial_policy_digest()
        assert isinstance(d, str) and len(d) == 16
        # the digest is stable
        assert wma_partial_policy_digest() == d

    def test_check_wma_policy(self):
        assert check_wma_partial_policy(WMA_PARTIAL_POLICY) == WMA_PARTIAL_POLICY
        with pytest.raises(ValueError):
            check_wma_partial_policy("FixedAgeWMA")


class TestRankMethodGate:
    def test_method_first_rejected_in_production(self):
        with pytest.raises(ValueError):
            check_rank_method("first", production=True, canonical="rank")

    def test_average_allowed(self):
        assert check_rank_method("average", production=True) == "average"

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            check_rank_method("bogus", production=True)
