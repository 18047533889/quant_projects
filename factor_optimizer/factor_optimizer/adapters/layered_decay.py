"""TRAIN-frozen research plan for observation-origin twenty-layer decay."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from factor_optimizer.adapters.repair_execution import _validate_frame


# QE's 20-quantile batch implementation materializes boundary arrays per date
# and full-panel sort/mask arrays. Bound its O(T*N) working set while retaining
# exact row-wise assignment semantics.
_QUANTILE_WORKING_BYTES = 64 * 1024 * 1024


def _assign_daily_quantiles(values, *, working_bytes=_QUANTILE_WORKING_BYTES):
    from quant_evaluator.metrics.quantile import assign_quantiles_batch

    if (isinstance(working_bytes, bool) or not isinstance(working_bytes, int)
            or working_bytes < 1):
        raise ValueError("working_bytes must be a positive integer")
    n_dates, n_assets = values.shape
    # QE's finite mask, filled panel, sorted panel and quantile result are
    # panel-sized; its 19-boundary arrays scale with dates, not assets.
    bytes_per_date = max(1, n_assets) * 40 + 20 * 1024
    if n_dates and bytes_per_date > working_bytes:
        raise ValueError("one date exceeds the layered-decay quantile working-set budget")
    rows_per_block = max(1, min(n_dates or 1, working_bytes // bytes_per_date))
    bins = np.empty(values.shape, dtype=np.int32)
    for start in range(0, n_dates, rows_per_block):
        stop = min(start + rows_per_block, n_dates)
        bins[start:stop] = assign_quantiles_batch(values[start:stop], n_quantiles=20)
    return bins


@dataclass(frozen=True)
class LayeredDecayPlan:
    half_lives: tuple
    training_context_ref: str

    def __post_init__(self):
        from factor_preprocess.transforms.layered_decay import layered_decay
        lives = tuple(self.half_lives)
        if len(lives) != 20:
            raise ValueError("exactly twenty layer half-lives required")
        layered_decay(np.empty((0, 0)), np.empty((0, 0), dtype=int), lives,
                      allow_research=True)
        if not isinstance(self.training_context_ref, str) or not self.training_context_ref.strip():
            raise ValueError("training_context_ref required")
        object.__setattr__(self, 'half_lives', tuple(float(h) for h in lives))

    family = 'DECAY_REFINEMENT'
    transform = 'layered_decay'

    @property
    def parameters(self):
        return (("half_lives", self.half_lives),)

    @property
    def identity(self):
        payload = ('layered-decay.v1', self.half_lives, self.training_context_ref)
        return hashlib.sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()

    def execute(self, values, *, allow_research=False):
        if allow_research is not True:
            raise ValueError("explicit research opt-in required")
        if isinstance(values, pd.DataFrame) and not values.columns.is_unique:
            raise ValueError("duplicate columns are ambiguous")
        frame = _validate_frame(values)
        if not all(g['date'].is_monotonic_increasing for _, g in frame.groupby('asset_id')):
            raise ValueError("per-asset dates must be monotone increasing")
        if frame.empty:
            return pd.Series(index=frame.index, dtype=float, name='value')
        from factor_preprocess.transforms.layered_decay import layered_decay
        panel = frame.pivot(index='date', columns='asset_id', values='value').sort_index()
        x = panel.to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        bins = _assign_daily_quantiles(x)
        output = layered_decay(x, bins, self.half_lives, allow_research=True)
        rows = panel.index.get_indexer(frame['date'])
        cols = panel.columns.get_indexer(frame['asset_id'])
        return pd.Series(output[rows, cols], index=frame.index, name='value')
