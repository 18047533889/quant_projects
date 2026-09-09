"""Deterministic V5 diagnosis routing with bounded, non-Cartesian trials."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import numpy as np
from typing import Mapping, Sequence, Tuple


class RecipeKind(str, Enum):
    FACTOR = "FACTOR_RECIPE"
    PORTFOLIO = "PORTFOLIO_RECIPE"


@dataclass(frozen=True)
class FrozenPortfolioRecipe:
    treatment: str
    parameters: Tuple[Tuple[str, float], ...]
    recipe_version: str = "portfolio-routing.v1"

    def __post_init__(self):
        frozen = tuple((str(k), float(v)) for k, v in self.parameters)
        if len({k for k, _ in frozen}) != len(frozen) or any(not np.isfinite(v) for _, v in frozen):
            raise ValueError("portfolio recipe parameters must be unique and finite")
        object.__setattr__(self, "parameters", frozen)
        expected = {"buy_hold_buffer": {"entry_rank", "exit_rank"},
                    "no_trade_band": {"target_rank", "band"},
                    "bottom_exclusion": {"minimum_rank"}}
        if self.treatment not in expected or {k for k, _ in frozen} != expected[self.treatment]:
            raise ValueError("portfolio recipe parameter set does not match treatment")
        p = dict(frozen)
        if self.treatment == "buy_hold_buffer" and not (0 <= p["exit_rank"] < p["entry_rank"] <= 1):
            raise ValueError("buy/hold ranks must satisfy 0 <= exit < entry <= 1")
        if self.treatment == "no_trade_band" and not (0 <= p["target_rank"] <= 1 and 0 <= p["band"] <= 1):
            raise ValueError("no-trade parameters outside [0,1]")
        if self.treatment == "bottom_exclusion" and not 0 <= p["minimum_rank"] <= 1:
            raise ValueError("minimum_rank outside [0,1]")

    @property
    def recipe_ref(self):
        raw = json.dumps({"treatment": self.treatment, "parameters": self.parameters,
                          "version": self.recipe_version}, sort_keys=True, separators=(",", ":"))
        return "portfolio-recipe:" + sha256(raw.encode()).hexdigest()

    def execute(self, signal_ranks: Sequence[Sequence[float]]) -> np.ndarray:
        """Compile the frozen recipe into stateful target inclusion weights."""
        x = np.asarray(signal_ranks, dtype=float)
        if x.ndim != 2 or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("portfolio recipe requires finite (T,N) ranks in [0,1]")
        p = dict(self.parameters); out = np.zeros_like(x); held = np.zeros(x.shape[1], dtype=bool)
        for t in range(x.shape[0]):
            if self.treatment == "buy_hold_buffer":
                held = (held & (x[t] >= p["exit_rank"])) | (x[t] >= p["entry_rank"])
            elif self.treatment == "no_trade_band":
                desired = x[t] >= p["target_rank"]
                change = np.abs(x[t] - p["target_rank"]) >= p["band"]
                held = np.where(change, desired, held)
            elif self.treatment == "bottom_exclusion":
                held = x[t] >= p["minimum_rank"]
            else:
                raise ValueError("unknown frozen portfolio treatment")
            if held.any(): out[t, held] = 1.0 / held.sum()
        return out


def compile_portfolio_recipe(treatment: str) -> FrozenPortfolioRecipe:
    parameters = {"buy_hold_buffer": (("entry_rank", .9), ("exit_rank", .8)),
                  "no_trade_band": (("target_rank", .9), ("band", .05)),
                  "bottom_exclusion": (("minimum_rank", .2),)}
    if treatment not in parameters:
        raise ValueError("treatment has no PortfolioRecipe compiler")
    return FrozenPortfolioRecipe(treatment, parameters[treatment])


def execute_routed_portfolio_trial(trial, signal_ranks, consumer, fidelity):
    """Resolve the frozen ref, execute weights, and pass them to execution/QE."""
    if trial.metadata.get("recipe_kind") != RecipeKind.PORTFOLIO.value:
        raise ValueError("trial is not a portfolio recipe")
    if not callable(consumer):
        raise TypeError("portfolio execution consumer must be callable")
    recipe = compile_portfolio_recipe(trial.metadata.get("treatment"))
    if trial.metadata.get("portfolio_recipe_ref") != recipe.recipe_ref:
        raise ValueError("portfolio recipe ref does not match compiled authority")
    return consumer(trial, recipe.execute(signal_ranks), fidelity)


@dataclass(frozen=True)
class RoutedTrial:
    trial_id: str
    diagnosis: str
    treatment: str
    recipe_kind: RecipeKind
    donor_family: str
    is_raw_baseline: bool = False
    routing_identity: str = ""

    def to_trial(self):
        """Compile the route into the canonical trial consumed by SearchRunner."""
        from factor_optimizer.contracts.trial import Trial, TrialStatus

        return Trial(
            trial_id=self.trial_id,
            mutation_id=self.trial_id,
            status=TrialStatus.PROPOSED,
            parent_factor_ids=[self.donor_family],
            metadata={
                "diagnosis": self.diagnosis,
                "treatment": self.treatment,
                "recipe_kind": self.recipe_kind.value,
                "donor_family": self.donor_family,
                "is_raw_baseline": self.is_raw_baseline,
                "routing_identity": self.routing_identity,
                "portfolio_recipe_ref": (compile_portfolio_recipe(self.treatment).recipe_ref
                                         if self.recipe_kind is RecipeKind.PORTFOLIO else None),
            },
        )


@dataclass(frozen=True)
class RoutingPolicy:
    max_new_candidates_initial: int = 8
    max_new_candidates_explicit: int = 12
    max_per_donor_family: int = 4
    policy_version: str = "V5.0"

    def __post_init__(self):
        for name, upper in (("max_new_candidates_initial", 8),
                            ("max_new_candidates_explicit", 12), ("max_per_donor_family", 12)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be a positive integer <= {upper}")
        if self.max_new_candidates_initial > self.max_new_candidates_explicit:
            raise ValueError("initial candidate cap cannot exceed explicit cap")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("policy_version is required")


_ROUTES: Mapping[str, Tuple[Tuple[str, RecipeKind], ...]] = {
    "HIGH_TURNOVER": (("causal_ewma_fast", RecipeKind.FACTOR),
                      ("causal_ewma_slow", RecipeKind.FACTOR),
                      ("buy_hold_buffer", RecipeKind.PORTFOLIO),
                      ("no_trade_band", RecipeKind.PORTFOLIO)),
    "STYLE_EXPOSURE": (("industry_neutral", RecipeKind.FACTOR),
                       ("industry_size_neutral", RecipeKind.FACTOR)),
    "SYMMETRIC_U": (("rank_abs_deviation", RecipeKind.FACTOR),
                    ("rank_squared_deviation", RecipeKind.FACTOR)),
    "ASYMMETRIC_U": (("left_hinge", RecipeKind.FACTOR),
                     ("right_hinge", RecipeKind.FACTOR)),
    "HEAVY_TAIL": (("robust_clip", RecipeKind.FACTOR),),
    "WINDOW_INSTABILITY": (("window_lower_neighbor", RecipeKind.FACTOR),
                           ("window_upper_neighbor", RecipeKind.FACTOR)),
    "BOTTOM_EXCLUSION": (("bottom_exclusion", RecipeKind.PORTFOLIO),),
}


def route_diagnoses(
    parent_factor_id: str,
    diagnoses: Sequence[str],
    *,
    policy: RoutingPolicy = RoutingPolicy(),
    explicit_budget: int | None = None,
) -> Tuple[RoutedTrial, ...]:
    """Return raw plus a deterministic union of single-treatment trials.

    Diagnoses are priority ordered. Treatments are never multiplied into a
    Cartesian product. An explicit expansion may raise the cap to 12, never
    beyond it; raw is retained outside the new-candidate count.
    """
    if not parent_factor_id:
        raise ValueError("parent_factor_id is required")
    cap = policy.max_new_candidates_initial
    if explicit_budget is not None:
        if isinstance(explicit_budget, bool) or not 1 <= explicit_budget <= policy.max_new_candidates_explicit:
            raise ValueError("explicit_budget must be in [1, 12]")
        cap = explicit_budget
    normalized = tuple(str(d).upper() for d in diagnoses)
    unknown = [d for d in normalized if d not in _ROUTES]
    if unknown:
        raise ValueError(f"unknown diagnoses: {unknown}")
    identity_payload = {"parent": parent_factor_id, "diagnoses": normalized, "policy": policy.__dict__,
                        "effective_budget": cap}
    routing_identity = sha256(json.dumps(identity_payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    output = [RoutedTrial(f"{parent_factor_id}:routing:{routing_identity}:raw", "BASELINE", "raw",
                          RecipeKind.FACTOR, parent_factor_id, True, routing_identity)]
    family_counts = {}
    seen = set()
    for diagnosis in normalized:
        token = str(diagnosis).upper()
        for treatment, kind in _ROUTES.get(token, ()):
            identity = (treatment, kind)
            if identity in seen:
                continue
            donor = parent_factor_id
            if family_counts.get(donor, 0) >= policy.max_per_donor_family or len(output)-1 >= cap:
                continue
            seen.add(identity)
            family_counts[donor] = family_counts.get(donor, 0) + 1
            output.append(RoutedTrial(f"{parent_factor_id}:routing:{routing_identity}:{kind.value}:{treatment}",
                                      token, treatment, kind, donor, False, routing_identity))
    return tuple(output)


__all__ = ["RecipeKind", "RoutedTrial", "RoutingPolicy", "route_diagnoses",
           "FrozenPortfolioRecipe", "compile_portfolio_recipe", "execute_routed_portfolio_trial"]
