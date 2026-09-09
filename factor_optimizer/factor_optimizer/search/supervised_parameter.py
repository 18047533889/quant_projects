"""Train-only selection and freezing for supervised repair parameters."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, Tuple


U_CENTER_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
TAIL_CUTOFF_GRID = (0.85, 0.90, 0.95)


@dataclass(frozen=True)
class FrozenSupervisedParameter:
    parent_factor_id: str
    repair_family: str
    parameter_name: str
    value: float
    candidate_grid: Tuple[float, ...]
    train_split_ref: str
    training_evidence_ref: str
    objective_id: str
    state_hash: str = ""

    def __post_init__(self):
        if self.repair_family not in {"U_SHAPE_REPAIR", "TAIL_SATURATION", "TAIL_HINGE"}:
            raise ValueError("unsupported supervised repair family")
        grid = tuple(float(v) for v in self.candidate_grid)
        if not grid or len(grid) != len(set(grid)) or not all(math.isfinite(v) for v in grid):
            raise ValueError("candidate_grid must be finite, unique, and non-empty")
        if self.value not in grid:
            raise ValueError("selected value must belong to the predeclared grid")
        for value in (self.parent_factor_id, self.parameter_name, self.train_split_ref,
                      self.training_evidence_ref, self.objective_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("frozen parameter identity fields are required")
        object.__setattr__(self, "candidate_grid", grid)
        payload = {
            "parent_factor_id": self.parent_factor_id, "repair_family": self.repair_family,
            "parameter_name": self.parameter_name, "value": self.value,
            "candidate_grid": grid, "train_split_ref": self.train_split_ref,
            "training_evidence_ref": self.training_evidence_ref, "objective_id": self.objective_id,
        }
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.state_hash and self.state_hash != expected:
            raise ValueError("frozen parameter state_hash mismatch")
        object.__setattr__(self, "state_hash", expected)

    def recipe_parameters(self, *, split_role: str) -> Mapping[str, float]:
        """Apply the frozen choice to validation/test without refitting."""
        if split_role not in {"TRAIN", "VALIDATION", "TEST"}:
            raise ValueError("unknown split role")
        return {self.parameter_name: self.value}


def fit_supervised_parameter(*, parent_factor_id: str, repair_family: str,
                             parameter_name: str, candidate_grid: Sequence[float],
                             train_scores: Mapping[float, float], split_role: str,
                             train_split_ref: str, training_evidence_ref: str,
                             objective_id: str) -> FrozenSupervisedParameter:
    """Choose from a finite grid using TRAIN evidence only."""
    if split_role != "TRAIN":
        raise ValueError("supervised repair parameters may only be fitted on TRAIN")
    grid = tuple(float(v) for v in candidate_grid)
    if set(train_scores) != set(grid):
        raise ValueError("training evidence must cover the complete predeclared grid")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in train_scores.values()):
        raise ValueError("all training objective values must be finite")
    # Stable tie-break: closest to the simplest baseline, then lower value.
    baseline = 0.5 if repair_family == "U_SHAPE_REPAIR" else max(grid)
    selected = min(grid, key=lambda value: (-float(train_scores[value]), abs(value - baseline), value))
    return FrozenSupervisedParameter(
        parent_factor_id, repair_family, parameter_name, selected, grid,
        train_split_ref, training_evidence_ref, objective_id,
    )
