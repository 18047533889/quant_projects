# -*- coding: utf-8 -*-
"""P3 pandas-only 边界：FFT/矩阵/随机/constant 等 intentionally 保留 pandas 路径。"""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

# 企业级 intentionally pandas-only（不做 Polars 移植）
INTENTIONALLY_PANDAS_ONLY = frozenset(
    {
        "constant",  # ScalarOperator
        "shuffle",
        "fft",
        "ifft",
        "convolve",
        "filter_bandpass",
        "filter_highpass",
        "filter_lowpass",
        "filter_notch",
        "wavelet",
        "wavelet_denoise",
        "mat_add",
        "mat_subtract",
        "mat_multiply",
        "mat_transpose",
        "mat_inverse",
        "mat_determinant",
        "mat_rank",
        "eig",
        "svd",
        "pca",
        "qr_decompose",
        "lu_decompose",
        "rand_exp",
        "rand_lognormal",
        "rand_normal",
        "rand_poisson",
        "rand_uniform",
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
    }
)


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def test_regress_slope_aggr_top_n_have_polars():
    for name in ("regress", "ridge", "lasso", "slope", "aggr_top_n", "ACF", "pacf", "norm"):
        assert "polars" in OperatorRegistry.backends_for(name), name


def test_pandas_only_is_subset_of_intentional_or_small():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    pandas_only = sorted(
        c for c in canon if "polars" not in OperatorRegistry.backends_for(c) and "pandas_numpy" in OperatorRegistry.backends_for(c)
    )
    unexpected = [c for c in pandas_only if c not in INTENTIONALLY_PANDAS_ONLY]
    assert len(pandas_only) <= 45, f"pandas-only 过多: {len(pandas_only)} {pandas_only}"
    assert len(unexpected) == 0, f"非 intentional pandas-only: {unexpected}"


def test_intentionally_pandas_only_still_have_pandas():
    for name in sorted(INTENTIONALLY_PANDAS_ONLY):
        if name not in OperatorRegistry.list_canonical():
            continue
        assert "pandas_numpy" in OperatorRegistry.backends_for(name), name
