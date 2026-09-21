"""TRAIN-frozen research plan for observation-origin twenty-layer decay."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from factor_optimizer.adapters.repair_execution import _validate_frame


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
        from quant_evaluator.metrics.quantile import assign_quantiles_batch
        from factor_preprocess.transforms.layered_decay import layered_decay
        panel = frame.pivot(index='date', columns='asset_id', values='value').sort_index()
        x = panel.to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        bins = assign_quantiles_batch(x, n_quantiles=20)
        output = layered_decay(x, bins, self.half_lives, allow_research=True)
        rows = panel.index.get_indexer(frame['date'])
        cols = panel.columns.get_indexer(frame['asset_id'])
        return pd.Series(output[rows, cols], index=frame.index, name='value')
