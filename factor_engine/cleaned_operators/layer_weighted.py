# -*- coding: utf-8 -*-
"""PIT-safe weighted cross-sectional primitives."""
from __future__ import annotations

from typing import Any
import numpy as np

from cleaned_operators.overhaul.base import EPS, Spec, aligned_pd, frame_pd, pl, pl_base_with, pl_cols, register_specs


def _stats(x, weight):
    valid = np.isfinite(x) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return None
    total = float(weight[valid].sum())
    if total <= EPS:
        return None
    mean = np.where(total) != 0, float(np.dot(x[valid], weight[valid]) / total), np.nan)
    variance = np.where(total) != 0, float(np.dot((x[valid] - mean) ** 2, weight[valid]) / total), np.nan)
    return mean, max(variance, 0.0)


def _weighted(x, weight, mode):
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        stats = _stats(x[row], weight[row])
        if stats is None:
            continue
        valid = np.isfinite(x[row]) & np.isfinite(weight[row]) & (weight[row] > 0)
        if mode == "mean":
            out[row, valid] = stats[0]
        elif mode == "demean":
            out[row, valid] = x[row, valid] - stats[0]
        elif stats[1] > EPS:
            out[row, valid] = np.where(np.sqrt(stats[1]) != 0, (x[row, valid] - stats[0]) / np.sqrt(stats[1]), np.nan)
    return out


def _group_weighted(x, group, weight, zscore):
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        labels: list[Any] = []
        for value in group[row]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if value not in labels:
                labels.append(value)
        for label in labels:
            members = np.asarray([value == label for value in group[row]], dtype=bool)
            stats = _stats(x[row, members], weight[row, members])
            if stats is None:
                continue
            local = np.isfinite(x[row, members]) & np.isfinite(weight[row, members]) & (weight[row, members] > 0)
            indices = np.flatnonzero(members)[local]
            if not zscore:
                out[row, indices] = stats[0]
            elif stats[1] > EPS:
                out[row, indices] = np.where(np.sqrt(stats[1]) != 0, (x[row, indices] - stats[0]) / np.sqrt(stats[1]), np.nan)
    return out


def pd_weighted(x, weight, mode):
    x, weight = aligned_pd(x, weight)
    return frame_pd(x, _weighted(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), mode))


def pd_group(x, group, weight, zscore):
    x, group, weight = aligned_pd(x, group, weight)
    return frame_pd(x, _group_weighted(x.to_numpy(dtype=float), group.to_numpy(dtype=object), weight.to_numpy(dtype=float), zscore))


def _pl_numeric(frame, cols):
    return np.column_stack([frame[c].cast(pl.Float64, strict=False).to_numpy() for c in cols])


def pl_weighted(x, weight, mode):
    cols = [c for c in pl_cols(x) if c in weight.columns]
    out = _weighted(_pl_numeric(x, cols), _pl_numeric(weight, cols), mode)
    return pl_base_with(x, {c: pl.Series(c, out[:, i]) for i, c in enumerate(cols)})


def pl_group(x, group, weight, zscore):
    cols = [c for c in pl_cols(x) if c in group.columns and c in weight.columns]
    groups = np.column_stack([np.asarray(group[c].to_list(), dtype=object) for c in cols])
    out = _group_weighted(_pl_numeric(x, cols), groups, _pl_numeric(weight, cols), zscore)
    return pl_base_with(x, {c: pl.Series(c, out[:, i]) for i, c in enumerate(cols)})


def register():
    register_specs({
        "cs_weighted_mean": Spec("cross_sectional", ["x", "weight"], "weighted cross-sectional mean", lambda x, w, **kw: pd_weighted(x, w, "mean"), lambda x, w, **kw: pl_weighted(x, w, "mean")),
        "cs_weighted_demean": Spec("cross_sectional", ["x", "weight"], "weighted cross-sectional demean", lambda x, w, **kw: pd_weighted(x, w, "demean"), lambda x, w, **kw: pl_weighted(x, w, "demean")),
        "cs_weighted_zscore": Spec("cross_sectional", ["x", "weight"], "weighted cross-sectional zscore", lambda x, w, **kw: pd_weighted(x, w, "zscore"), lambda x, w, **kw: pl_weighted(x, w, "zscore")),
        "group_weighted_mean": Spec("cross_sectional", ["x", "group", "weight"], "within-group weighted mean", lambda x, g, w, **kw: pd_group(x, g, w, False), lambda x, g, w, **kw: pl_group(x, g, w, False)),
        "group_weighted_zscore": Spec("cross_sectional", ["x", "group", "weight"], "within-group weighted zscore", lambda x, g, w, **kw: pd_group(x, g, w, True), lambda x, g, w, **kw: pl_group(x, g, w, True)),
    })


register()
