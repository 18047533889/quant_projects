"""Shared research NumPy recurrence for dense and sparse layered decay."""
from __future__ import annotations

import numbers
import numpy as np


class LayeredDecayState:
    """Per-asset observation-origin state; intentionally a NumPy research kernel."""

    def __init__(self, n_assets: int, half_lives):
        lives = tuple(half_lives)
        if (isinstance(n_assets, bool) or not isinstance(n_assets, int) or n_assets < 0
                or not 1 <= len(lives) <= 20
                or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, numbers.Real)
                       or not np.isfinite(h) or not 1 <= h <= 60 for h in lives)):
            raise ValueError("invalid layered-decay state dimensions or half-lives")
        self.alpha = -np.expm1(-np.log(2.) / np.asarray(lives, dtype=float))
        self.rho = 1. - self.alpha
        self.numerator = np.zeros((n_assets, len(self.alpha)))
        self.mass = np.zeros_like(self.numerator)
        self.scale = np.zeros(n_assets)
        self.active_assets = np.empty(0, dtype=np.intp)
        self._last_seen = np.zeros(n_assets, dtype=np.int64)
        self._epoch = 0

    def step(self, values, layers, *, asset_codes=None):
        """Consume a dense date row and return its lagged output by asset."""
        values = np.asarray(values, dtype=float)
        layers = np.asarray(layers)
        if asset_codes is not None:
            return self.step_sparse(values, layers, asset_codes)
        if (values.shape != (len(self.scale),) or layers.shape != values.shape
                or layers.dtype.kind not in "iu" or np.any(layers < -1)
                or np.any(layers >= len(self.alpha))):
            raise ValueError("dense layered-decay rows must match the state asset count")
        good = np.flatnonzero(np.isfinite(values) & (layers >= 0))
        if len(good) == len(self.scale) and len(self.active_assets) == len(self.scale):
            # Full, continuously observed panel rows need no code validation,
            # sorting, or active-set intersection.
            self.numerator *= self.rho
            self.mass *= self.rho
            return self._apply_inputs(values, layers, good)
        return self.step_sparse(values[good], layers[good], good)

    def step_sparse(self, values, layers, asset_codes):
        """Update only prior/current active assets; codes and state are sorted."""
        values = np.asarray(values, dtype=float)
        layers = np.asarray(layers)
        codes = np.asarray(asset_codes)
        if (values.ndim != 1 or layers.shape != values.shape or codes.shape != values.shape
                or codes.dtype.kind not in "iu" or layers.dtype.kind not in "iu"
                or np.any(layers < -1) or np.any(layers >= len(self.alpha))
                or np.any(codes < 0) or np.any(codes >= len(self.scale))
                or len(np.unique(codes)) != len(codes)):
            raise ValueError("sparse layered-decay rows require unique valid asset codes")
        good_rows = np.flatnonzero(np.isfinite(values) & (layers >= 0))
        return self._step_sparse_trusted(values[good_rows], layers[good_rows], codes[good_rows])

    def _step_sparse_trusted(self, source, layer, assets):
        # Input identities and uniqueness are guaranteed by the validated adapter.
        output = np.full(len(self.scale), np.nan)
        self._epoch += 1
        self._last_seen[assets] = self._epoch
        if len(self.active_assets):
            retained_mask = self._last_seen[self.active_assets] == self._epoch
            retained = self.active_assets[retained_mask]
            dropped = self.active_assets[~retained_mask]
            self.numerator[retained] *= self.rho
            self.mass[retained] *= self.rho
            self.numerator[dropped] = 0.
            self.mass[dropped] = 0.
            self.scale[dropped] = 0.
        if not len(assets):
            self.active_assets = assets
            return output
        output[assets] = self._apply_inputs(source, layer, assets)
        self.active_assets = assets
        return output

    def _apply_inputs(self, source, layer, assets):
        new_scale = np.maximum(self.scale[assets], np.abs(source))
        ratio = np.divide(self.scale[assets], new_scale, out=np.zeros(len(assets)), where=new_scale > 0)
        self.numerator[assets] *= ratio[:, None]
        self.scale[assets] = new_scale
        normalized = np.divide(source, new_scale, out=np.zeros(len(assets)), where=new_scale > 0)
        selected = layer
        self.numerator[assets, selected] += self.alpha[selected] * normalized
        self.mass[assets, selected] += self.alpha[selected]
        weighted = self.numerator[assets].sum(axis=1) / self.mass[assets].sum(axis=1)
        return np.clip(weighted, -1., 1.) * new_scale
