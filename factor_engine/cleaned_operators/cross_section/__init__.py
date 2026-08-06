# -*- coding: utf-8 -*-
"""Next-stage cross-sectional operators (2026-08 expansion).

Robust cross-sectional residuals, peer / industry relative-value operators and
rolling panel model operators (PCA / PLS / PCR / ElasticNet / regime / MoE /
autoencoder anomaly).  All output one scalar per (TradeDate, Symbol).
"""
from __future__ import annotations

__all__ = ["robust_cs", "peer_ops", "panel_model"]
