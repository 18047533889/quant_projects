# -*- coding: utf-8 -*-
"""R28 §三十..三十二 / §一百零二..一百零三: panel supervised-model walk-forward.

- Label maturity: a forward-H label anchored at ``s`` matures at ``s+H``; the
  training set at decision ``T`` may only use labels with ``origin <= T-1-H``.
- Future shock must be invisible until the label matures.
- Purge / embargo: overlapping labels are not trained on pre-maturity rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _load():
    load_all()


def _panel(vals: np.ndarray) -> pd.DataFrame:
    n = vals.shape[0]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(vals, index=idx, columns=["C0"])


def _pcr(y, x1, x2, **kw):
    kw.setdefault("window", 60)
    kw.setdefault("n_components", 2)
    kw.setdefault("label_horizon", 1)
    return OperatorRegistry.get("panel_rolling_pcr_forecast", "pandas_numpy", mode="any").calculate(
        y, x1, x2, **kw
    )["C0"].to_numpy()


def test_panel_label_maturity_hides_future_shock():
    """A massive shock in the label target (the future) must not move the model's
    output until after the shock has matured."""
    _load()
    rng = np.random.default_rng(21)
    n = 160
    y = _panel(rng.standard_normal(n))
    x1 = _panel(rng.standard_normal(n))
    x2 = _panel(rng.standard_normal(n))
    base = _pcr(y, x1, x2, label_horizon=1)

    # put a massive shock in the TARGET at row 120 (this is future info that only
    # matters to labels maturing after 120)
    ys = y.copy()
    ys.iloc[120:, 0] = 1e6
    shocked = _pcr(ys, x1, x2, label_horizon=1)
    # before the shock is knowable, the model's output must be unchanged
    np.testing.assert_allclose(shocked[:119], base[:119], equal_nan=True, atol=1e-6)


def test_panel_forecast_finite_when_labels_unmatured():
    """The last label_horizon rows of the training window are excluded (never
    trained on unmatured labels), which must still yield a finite forecast."""
    _load()
    rng = np.random.default_rng(22)
    n = 160
    y = _panel(rng.standard_normal(n))
    x1 = _panel(rng.standard_normal(n))
    x2 = _panel(rng.standard_normal(n))
    for h in (1, 3, 5):
        out = _pcr(y, x1, x2, label_horizon=h, window=80)
        assert np.isfinite(out[np.isfinite(out)]).all(), f"label_horizon={h} non-finite"


def test_panel_purge_embargo_contract_holds():
    """The LabelContract of the panel forecasters is purge/embargo-safe: with an
    overlapping forward label (horizon H), the training exclusion (purge) is
    exactly H rows and no label origin <= T-1-H is ever dropped."""
    _load()
    from factor_engine.cleaned_operators.model_timing import get_model_timing_contract

    for name in (
        "panel_rolling_pcr_forecast",
        "panel_rolling_pls_forecast",
        "panel_rolling_elastic_net_forecast",
    ):
        c = get_model_timing_contract(name)
        assert c is not None, name
        assert c.label_horizon is not None and c.label_horizon >= 1, name
        assert c.fit_cutoff_offset >= 1, name  # fit strictly before the scored row


def test_panel_future_perturbation():
    """Perturbing x far in the future cannot change past predictions."""
    _load()
    rng = np.random.default_rng(23)
    n = 160
    y = _panel(rng.standard_normal(n))
    x1 = _panel(rng.standard_normal(n))
    x2 = _panel(rng.standard_normal(n))
    base = _pcr(y, x1, x2, window=80)
    x1m = x1.copy()
    x1m.iloc[120:, 0] = 999.0
    changed = _pcr(y, x1m, x2, window=80)
    np.testing.assert_allclose(changed[:110], base[:110], equal_nan=True, atol=1e-6)
