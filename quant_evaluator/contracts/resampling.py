"""QE-owned, reproducible original-clock moving-block plan (not a score)."""
from dataclasses import dataclass
import numpy as np
from ._hashutil import stable_content_hex
from .metric_artifacts import _freeze_array


@dataclass(frozen=True)
class ResamplingPlan:
    time_ids: tuple
    clock_ref: str
    block_length: int
    num_replicates: int
    seed: int
    segment_ids: tuple = ()
    algorithm: str = "moving_block_original_clock.v1"

    def __post_init__(self):
        object.__setattr__(self, 'time_ids', tuple(self.time_ids))
        object.__setattr__(self, 'segment_ids', tuple(self.segment_ids))
        if not self.clock_ref or len(set(self.time_ids)) != len(self.time_ids):
            raise ValueError('unique time ids and explicit clock required')
        for name, minimum in (('block_length', 1), ('num_replicates', 2), ('seed', 0)):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < minimum:
                raise ValueError(f'{name} must be integer >= {minimum}')
        if len(self.time_ids) < 2*self.block_length:
            raise ValueError('insufficient time span for moving blocks')
        if self.segment_ids and len(self.segment_ids) != len(self.time_ids):
            raise ValueError('segment axis mismatch')
        if self.algorithm != 'moving_block_original_clock.v1':
            raise ValueError('unsupported resampling algorithm')
        if not self.valid_starts:
            raise ValueError('no legal within-segment blocks')

    @property
    def valid_starts(self):
        return tuple(i for i in range(len(self.time_ids)-self.block_length+1)
                     if not self.segment_ids or len(set(self.segment_ids[i:i+self.block_length])) == 1)

    @property
    def content_hash(self):
        return stable_content_hex(tag=self.algorithm, fields={"time_ids": self.time_ids, "clock_ref": self.clock_ref,
                                   "block_length": self.block_length, "num_replicates": self.num_replicates, "seed": self.seed, "segment_ids": self.segment_ids})

    @property
    def replicate_ids(self):
        return tuple(f'{self.content_hash}:{i}' for i in range(self.num_replicates))

    def indices(self):
        n = len(self.time_ids)
        rng = np.random.default_rng(self.seed)
        starts = rng.choice(self.valid_starts, size=(self.num_replicates, (n+self.block_length-1)//self.block_length))
        return _freeze_array((starts[..., None] + np.arange(self.block_length)).reshape(self.num_replicates, -1)[:, :n], 'resampling indices')
