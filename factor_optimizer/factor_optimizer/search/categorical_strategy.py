"""Categorical search strategy (TPE-style) for treatment auto-optimization.

The runner's numeric-kernel strategies (Bayesian/TPE) reject categorical
``choice`` parameters.  Factor auto-treatment search, however, is inherently
categorical: a treatment *recipe* is a discrete choice among variants, each
with its own parameter ranges.  This strategy searches over that categorical
grammar WITHOUT forcing categorical values to floats.

Design:
  - ``CategoricalRecipe`` describes one discrete treatment recipe: a name plus
    per-parameter discrete ranges (param -> list of candidate values).
  - ``CategoricalSearchStrategy`` (a ``SearchStrategy``) does a short
    quasi-random warmup that covers every recipe, then a simple TPE-style
    density model over the good-utility vs bad-utility split of past trials to
    propose the next candidate recipe and sample its parameters.
  - The proposed candidate keeps its categorical identity (recipe name) and
    parameter values verbatim — nothing is coerced to a float index.
"""

import json
import math
import numbers
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.strategies import (
    SearchStrategy,
    SearchStrategyState,
    StrategyContext,
    _capture_python_rng,
    _from_jsonable,
    _jsonable,
    _restore_python_rng,
)


@dataclass(frozen=True)
class CategoricalRecipe:
    """One discrete treatment-recipe candidate in the categorical space.

    Attributes:
        name: Unique recipe identifier (a plain string, NOT coerced to a
            numeric index).
        param_ranges: ``{param_name: [discrete values...]}``.  Each value is a
            candidate the recipe may take; sampling picks uniformly.  Values
            may be any JSON-safe type (strings, numbers) and are returned
            verbatim.
    """

    name: str
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("recipe name must be a non-empty string")
        if not isinstance(self.param_ranges, dict):
            raise ValueError("param_ranges must be a dict")
        for param, choices in self.param_ranges.items():
            if not isinstance(param, str) or not param.strip():
                raise ValueError("parameter names must be non-empty strings")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"recipe '{self.name}' parameter '{param}' must have a "
                    "non-empty choices list"
                )
            # Keep the values verbatim (no float coercion).  JSON-safety check
            # so checkpoint/resume never breaks.
            json.dumps(choices)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "param_ranges": self.param_ranges}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalRecipe":
        return cls(name=data["name"], param_ranges=dict(data["param_ranges"]))


def _sample(rng: random.Random, choices: List[Any]) -> Any:
    """Uniform sample from a discrete choice list (keeps original type)."""
    return choices[rng.randrange(len(choices))]


class CategoricalSearchStrategy(SearchStrategy):
    """TPE-style search over a categorical treatment grammar.

    Parameters:
        recipes: Ordered list of ``CategoricalRecipe`` candidates.
        n_initial: Number of warmup (quasi-random) proposals before the
            TPE-style density model takes over.
        gamma: Fraction of the recorded (highest-utility) trials treated as
            the "good" partition.
        seed: RNG seed for reproducibility.
        objective_spec / context: single direction authority (inherited from
            the base strategy; the spec is authoritative).
    """

    def __init__(
        self,
        recipes: Sequence[CategoricalRecipe],
        n_initial: int = 5,
        gamma: float = 0.25,
        seed: Optional[int] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ) -> None:
        if not isinstance(recipes, (list, tuple)) or not recipes:
            raise ValueError("categorical strategy requires a non-empty recipe list")
        if not all(isinstance(r, CategoricalRecipe) for r in recipes):
            raise TypeError("recipes must be CategoricalRecipe instances")
        names = [r.name for r in recipes]
        if len(names) != len(set(names)):
            raise ValueError("recipe names must be unique")
        if not isinstance(n_initial, int) or isinstance(n_initial, bool) or n_initial < 0:
            raise ValueError("n_initial must be a non-negative integer")
        if not isinstance(gamma, numbers.Real) or isinstance(gamma, bool):
            raise ValueError("gamma must be a number in (0, 1]")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be a number in (0, 1]")
        # Build a SearchSpace so the base SearchStrategy constructor works, but
        # the strategy NEVER uses numeric kernels; the categorical grammar is
        # handled natively.  A minimal dummy float parameter satisfies the base
        # contract without forcing categorical values to floats.
        from factor_optimizer.search.strategies import ParameterSpace, SearchSpace
        self._recipes = list(recipes)
        self._recipe_map = {r.name: r for r in recipes}
        # Per-recipe params are stored in each CategoricalRecipe; base space is
        # only a placeholder satisfying the base constructor (unused for
        # proposals).
        super().__init__(
            SearchSpace(parameters=[ParameterSpace("__probe__", "float", 0.0, 1.0)]),
            seed=seed,
            objective_spec=objective_spec,
            context=context,
        )
        self.n_initial = n_initial
        self.gamma = gamma
        # history of (recipe_name, params, utility)
        self._cat_history: List[Tuple[str, Dict[str, Any], float]] = []
        self.np_rng = np.random.default_rng(seed)
        self._random = random.Random(seed)
        # warmup: a shuffled queue visiting every recipe once, then cycling.
        self._warmup_queue: List[str] = []
        self._rebuild_warmup_queue()

    # -- private helpers ------------------------------------------------------

    def _rebuild_warmup_queue(self) -> None:
        queue = list(self._recipe_map.keys())
        self._random.shuffle(queue)
        self._warmup_queue = queue

    def _draw_params(self, recipe: CategoricalRecipe) -> Dict[str, Any]:
        return {
            p: _sample(self._random, choices)
            for p, choices in recipe.param_ranges.items()
        }
    def _propose_recipe_dict(self) -> Dict[str, Any]:
        """Choose a candidate recipe (and sample its params).

        Warmup: quasi-random coverage of every recipe.  Then a simple TPE-style
        density split over recorded (recipe, utility) pairs.
        """
        if len(self._cat_history) < self.n_initial:
            if not self._warmup_queue:
                self._rebuild_warmup_queue()
            recipe_name = self._warmup_queue.pop()
            recipe = self._recipe_map[recipe_name]
            params = self._draw_params(recipe)
            return {"recipe": recipe_name, "params": params}

        # TPE-style: split history into good/bad by utility.
        sorted_hist = sorted(self._cat_history, key=lambda row: row[2], reverse=True)
        n_good = max(1, int(math.ceil(self.gamma * len(sorted_hist))))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:]
        good_counts: Dict[str, int] = {}
        bad_counts: Dict[str, int] = {}
        for rn, _, _ in good:
            good_counts[rn] = good_counts.get(rn, 0) + 1
        for rn, _, _ in bad:
            bad_counts[rn] = bad_counts.get(rn, 0) + 1

        # Score each recipe by l(r)/g(r) (likelihood-ratio style), preferring
        # recipes more common in the good partition.
        best_recipe: Optional[str] = None
        best_ratio: float = -math.inf
        eps = 1e-6
        for rn in self._recipe_map:
            l = good_counts.get(rn, 0) + eps
            g = bad_counts.get(rn, 0) + eps
            ratio = l / g
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = rn
        recipe = self._recipe_map[best_name]
        params = self._draw_params(recipe)
        return {"recipe": recipe.name, "params": params}

    def propose_recipe(self) -> Dict[str, Any]:
        """Return the next candidate as a plain dict (no Trial wrapper).

        Example: ``{"recipe": "winsorize", "params": {"k": 3, "clip": "1.5"}}``.
        """
        return self._propose_recipe_dict()

    # -- SearchStrategy interface -------------------------------------------

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        """Suggest a new trial with categorical params in ``metadata["params"]``.

        ``params`` carries ``{"recipe": <str>, ...recipe params...}`` so the
        categorical identity is preserved verbatim (never coerced to floats).
        """
        cand = self._propose_recipe_dict()
        params = dict(cand["params"])
        params["recipe"] = cand["recipe"]
        self._next_proposal_index()
        return Trial(
            trial_id=trial_id or self._make_trial_id("categorical", params),
            mutation_id="categorical-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "categorical"},
        )

    def record(self, params: Dict[str, Any], score: float) -> None:
        """Record a completed evaluation (the strategy learns from history)."""
        super().record(params, score)
        recipe = params.get("recipe")
        if recipe is None:
            # Tolerate a params dict without an explicit recipe key by not
            # treating it as a categorical observation (recorded raw only).
            return
        utility = self._context.to_utility(float(score))
        self._cat_history.append((str(recipe), dict(params), utility))

    # -- checkpoint / resume -------------------------------------------------

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "recipes": [r.to_dict() for r in self._recipes],
                "n_initial": self.n_initial,
                "gamma": self.gamma,
                "seed": self.seed,
                "context": self._context.to_dict(),
            },
            "cat_history": [[rn, p, u] for (rn, p, u) in self._cat_history],
            "warmup_queue": list(self._warmup_queue),
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        self._cat_history = [
            (rn, dict(p), float(u))
            for (rn, p, u) in fields.get("cat_history", [])
        ]
        self._warmup_queue = list(fields.get("warmup_queue", []))
        self.np_rng = np.random.default_rng(self.seed)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self.save_state()))

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "CategoricalSearchStrategy":
        """Deserialize a standalone categorical strategy checkpoint.

        The base dispatcher only knows numeric strategies; this override lets a
        ``CategoricalSearchStrategy`` round-trip its own state without editing
        the shared registry.
        """
        raw = dict(_from_jsonable(dict(data)))
        fields = raw.get("strategy_fields", {})
        ctor = fields.get("ctor", {})
        recipes = [CategoricalRecipe.from_dict(r) for r in ctor.get("recipes", [])]
        kwargs = {
            "recipes": recipes,
            "n_initial": ctor.get("n_initial", 5),
            "gamma": ctor.get("gamma", 0.25),
            "seed": ctor.get("seed"),
        }
        if "context" in ctor:
            kwargs["context"] = StrategyContext.from_dict(ctor["context"])
        strategy = cls(**kwargs)
        strategy.load_state(SearchStrategyState(**raw))
        return strategy

    def _extra_rng_states(self) -> Dict[str, Dict[str, Any]]:
        return {"_random": _capture_python_rng(self._random)}

    def _restore_extra_rng_states(self, states: Dict[str, Dict[str, Any]]) -> None:
        if "_random" in states:
            _restore_python_rng(self._random, states["_random"])
