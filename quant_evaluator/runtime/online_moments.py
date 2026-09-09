"""Mergeable centered moments for finite scalar observations (sample variance)."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class OnlineMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    positive: int = 0
    origin: float = 0.0
    mean_offset: float = 0.0

    def __post_init__(self):
        # Lossless promotion of earlier centered state; never reinterpret its
        # absolute mean as an offset from zero during the next merge.
        if self.count and self.origin == 0.0 and self.mean_offset == 0.0 and self.mean != 0.0:
            self.origin = self.mean

    def update(self, values):
        data = np.asarray(values, dtype=np.float64).reshape(-1)
        data = data[np.isfinite(data)]
        if not data.size:
            return
        # Offset summation avoids loss of small spread under a large location.
        anchor = data[0]
        centered = data - anchor
        average = float(np.mean(centered))
        other = OnlineMoments(int(data.size), float(anchor + average),
                              float(np.sum((centered-average)**2)), int(np.count_nonzero(data > 0)),
                              float(anchor), average)
        self.merge(other)

    def merge(self, other):
        if not other.count:
            return
        if not self.count:
            self.count, self.mean, self.m2, self.positive = other.count, other.mean, other.m2, other.positive
            self.origin, self.mean_offset = other.origin, other.mean_offset
            return
        total = self.count + other.count
        delta = (other.origin-self.origin) + other.mean_offset-self.mean_offset
        self.m2 += other.m2 + delta*delta*self.count*other.count/total
        self.mean_offset += delta*other.count/total
        self.mean = self.origin + self.mean_offset
        self.count = total
        self.positive += other.positive

    def summary(self):
        std = math.sqrt(max(0.0, self.m2)/(self.count-1)) if self.count > 1 else float("nan")
        return {"count": self.count, "mean": self.mean if self.count else float("nan"),
                "std": std, "raw_icir": self.mean/std if math.isfinite(std) and std > 0 else float("nan"),
                "positive_ratio": self.positive/self.count if self.count else float("nan")}
