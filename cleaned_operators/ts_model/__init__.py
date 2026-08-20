# -*- coding: utf-8 -*-
"""Next-stage time-series model operators (2026-08 expansion).

Rolling regression, AR / mean-reversion, state-space (Kalman), volatility
(GARCH/HAR), complexity / structural-break, wavelet / spectral, matrix-profile
sequence anomaly, and path-signature kernels.  All output one scalar per
(TradeDate, Symbol); expensive model kernels are tagged with high cost and stay
``experimental``.
"""
from __future__ import annotations

__all__ = [
    "dynamic_regression",
    "ar_meanrev",
    "state_space",
    "volatility",
    "complexity",
    "wavelet_spectral",
    "sequence_anomaly",
    "path_signature",
]
