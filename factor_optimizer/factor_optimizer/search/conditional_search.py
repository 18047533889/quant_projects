"""Hierarchical conditional search over (repair-family, parameters) (R61-FI-035).

Plan §24 upgrades the recipe-only TPE in
:mod:`factor_optimizer.search.categorical_strategy` from modelling only

    p(repair_family | good)

to modelling the joint conditional distribution

    p(repair_family, parameters | good)

where the *parameter domain is bound to the repair family* (a family's
parameters only make sense when that family is active) and every family's
parameter candidates are drawn from a **prior** rather than from a flat
uniform grid.

The planner's three worked examples are the canonical conditional trees:

- ``CAUSAL_SMOOTHING`` → ``method`` ∈ {EWMA, KAMA, IIR, Kalman} +
  ``natural_time_scale_relative`` ∈ (0, ∞);
- ``U_SHAPE_REPAIR`` → ``center`` / ``power`` / optional ``asymmetry``;
- ``NEUTRALIZATION`` (any of INDUSTRY_/SIZE_/STYLE_NEUTRALIZATION) →
  ``exposure_set`` + ``method``.

This module extends the existing recipe search without replacing the
scheduler: it keeps the runner/strategy interface contract
(``propose``/``record``/``save_state``/``load_state``) and adds a
conditional model over the family×parameter space.

Prior sources (plan §24) are injectable through the
:class:`ConditionalPriorProvider` Protocol; a deterministic stub default
(``StaticPriorProvider``) resolves priors from the versioned repair-family
declaration registry (FI-034), so FO contains no second prior authority.
"""

from __future__ import annotations

import math
import numbers
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.policy.repair import DiagnosisKind
from factor_optimizer.policy.repair_registry import (
    ParameterPrior,
    ParameterSchema,
    RepairFamily,
    RepairFamilyDeclaration,
    RepairFamilyRegistry,
)
from factor_optimizer.search.strategies import (
    SearchStrategy,
    SearchStrategyState,
    StrategyContext,
    _capture_python_rng,
    _from_jsonable,
    _jsonable,
    _restore_python_rng,
)


# ---------------------------------------------------------------------------
# Conditional parameter domain (bound to a repair family)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalParameter:
    """One parameter of a repair family with a bounded conditional domain.

    Attributes:
        name: Parameter name (e.g. ``method``, ``natural_time_scale_relative``).
        kind: ``"choice"`` (discrete candidates) or ``"float"``/``"int"``
            (bounded numeric).
        choices: Discrete candidates for ``kind == "choice"``.
        low/high: Inclusive numeric bounds for float/int parameters.
        prior: Default anchor value (may be None → sample uniformly).
    """

    name: str
    kind: str
    choices: Tuple[Any, ...] = ()
    low: Optional[float] = None
    high: Optional[float] = None
    prior: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        if self.kind not in ("choice", "float", "int"):
            raise ValueError(
                f"unknown parameter kind {self.kind!r} for {self.name!r} "
                "(must be 'choice', 'float' or 'int')"
            )
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(
                    f"choice parameter {self.name!r} requires non-empty choices"
                )
            object.__setattr__(self, "choices", tuple(self.choices))
        else:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"numeric parameter {self.name!r} requires low and high"
                )
            if isinstance(self.low, bool) or not isinstance(self.low, numbers.Real):
                raise ValueError(f"{self.name!r} low must be a number")
            if isinstance(self.high, bool) or not isinstance(self.high, numbers.Real):
                raise ValueError(f"{self.name!r} high must be a number")
            if self.low >= self.high:
                raise ValueError(
                    f"numeric parameter {self.name!r} requires low < high"
                )
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))
        if self.prior is not None:
            # Prior anchor must be inside the declared domain (fail closed).
            if self.kind == "choice":
                if self.prior not in self.choices:
                    raise ValueError(
                        f"prior {self.prior!r} for choice parameter {self.name!r} "
                        f"is not one of {self.choices}"
                    )
            else:
                if isinstance(self.prior, bool) or not isinstance(
                    self.prior, numbers.Real
                ):
                    raise ValueError(
                        f"prior for numeric parameter {self.name!r} must be a number"
                    )
                if not self.low <= float(self.prior) <= self.high:
                    raise ValueError(
                        f"prior {self.prior!r} for parameter {self.name!r} "
                        f"outside [{self.low}, {self.high}]"
                    )
                object.__setattr__(self, "prior", float(self.prior))

    def sample(self, rng: random.Random) -> Any:
        """Draw one value from this parameter's conditional domain."""
        if self.kind == "choice":
            return self.choices[rng.randrange(len(self.choices))]
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(self.low, self.high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "choices": list(self.choices),
            "low": self.low,
            "high": self.high,
            "prior": self.prior,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalParameter":
        return cls(**dict(data))


@dataclass(frozen=True)
class ConditionalFamily:
    """A repair family's conditional parameter tree (plan §24).

    Attributes:
        family: The repair family name (upper-snake, e.g. ``CAUSAL_SMOOTHING``).
        parameters: Ordered conditional parameters bound to this family.  A
            candidate for this family MUST carry exactly these parameters.
    """

    family: str
    parameters: Tuple[ConditionalParameter, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("family must be a non-empty string")
        if self.family not in RepairFamily.names():
            raise ValueError(
                f"unknown repair family {self.family!r}; expected one of "
                f"{RepairFamily.names()}"
            )
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(
                f"conditional family {self.family!r} has duplicate parameters"
            )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def validate_params(self, params: Mapping[str, Any]) -> None:
        """Fail closed when *params* does not exactly match this family's tree."""
        expected = set(self.parameter_names)
        actual = set(params)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            detail = []
            if missing:
                detail.append(f"missing: {missing}")
            if extra:
                detail.append(f"unexpected: {extra}")
            raise ValueError(
                f"params for family {self.family!r} do not match its "
                "conditional domain: " + "; ".join(detail)
            )
        by_name = {p.name: p for p in self.parameters}
        for name, value in params.items():
            param = by_name[name]
            if param.kind == "choice":
                if value not in param.choices:
                    raise ValueError(
                        f"value {value!r} for choice param {name!r} of "
                        f"{self.family!r} not in {param.choices}"
                    )
            elif param.kind == "int":
                if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                        or float(value) != int(value):
                    raise ValueError(
                        f"value {value!r} for int param {name!r} must be integral"
                    )
                if not param.low <= float(value) <= param.high:
                    raise ValueError(
                        f"value {value!r} for int param {name!r} out of "
                        f"[{param.low}, {param.high}]"
                    )
            else:
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise ValueError(
                        f"value for float param {name!r} must be a number"
                    )
                if not param.low <= float(value) <= param.high:
                    raise ValueError(
                        f"value {value!r} for float param {name!r} out of "
                        f"[{param.low}, {param.high}]"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "parameters": [p.to_dict() for p in self.parameters]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalFamily":
        return cls(
            family=data["family"],
            parameters=tuple(ConditionalParameter.from_dict(p) for p in data["parameters"]),
        )


def _parse_schema(
    declaration: RepairFamilyDeclaration,
) -> Dict[str, ConditionalParameter]:
    """Compile a family declaration's schema/prior into ConditionalParameters."""
    schema: ParameterSchema = declaration.parameter_schema
    prior: ParameterPrior = declaration.parameter_prior
    params: Dict[str, ConditionalParameter] = {}
    for name, spec in schema.params.items():
        if spec.startswith("choice:"):
            candidates = tuple(spec.split(":", 1)[1].split("|"))
            params[name] = ConditionalParameter(
                name=name, kind="choice", choices=candidates,
                prior=prior.value_of(name),
            )
        elif spec.startswith("float:"):
            _, low, high = spec.split(":")
            params[name] = ConditionalParameter(
                name=name, kind="float", low=float(low), high=float(high),
                prior=prior.value_of(name),
            )
        elif spec.startswith("int:"):
            _, low, high = spec.split(":")
            params[name] = ConditionalParameter(
                name=name, kind="int", low=int(low), high=int(high),
                prior=prior.value_of(name),
            )
        elif spec == "flag":
            params[name] = ConditionalParameter(
                name=name, kind="choice", choices=(True, False),
                prior=prior.value_of(name),
            )
        else:  # pragma: no cover - guarded by ParameterSchema validation
            raise ValueError(f"unknown schema spec {spec!r}")
    return params


def build_conditional_tree(
    families: Sequence[RepairFamily | str],
    registry: Optional[RepairFamilyRegistry] = None,
) -> Tuple[ConditionalFamily, ...]:
    """Compile the family declarations into the conditional search tree.

    The tree binds every declared parameter domain + prior of each family
    (plan §24).  The default registry (FI-034) provides the canonical 20-family
    tree; an injected registry lets tests build arbitrary conditional trees
    over the same vocabulary.
    """
    registry = registry or RepairFamilyRegistry.default()
    out: List[ConditionalFamily] = []
    for family in families:
        name = family.name if isinstance(family, RepairFamily) else family
        declaration = registry.get(name)
        compiled = _parse_schema(declaration)
        out.append(
            ConditionalFamily(
                family=name,
                parameters=tuple(compiled[n] for n in declaration.parameter_schema.params),
            )
        )
    return tuple(out)


class ConditionalPriorProvider(Protocol):
    """Injected prior source for conditional parameter values (plan §24).

    A provider resolves per-family parameter prior values from richer
    sources — factor natural horizon / existing DSL parameter values /
    operator metadata / diagnosis evidence — when available.  It returns
    ``None`` for a parameter it has no opinion about (fall back to the
    declared static prior, then to the uniform domain).
    """

    def prior_for(self, family: str, parameter: str) -> Optional[Any]:
        """Return a JSON-safe prior value for *parameter* of *family* (or None)."""
        ...


@dataclass(frozen=True)
class StaticPriorProvider:
    """Deterministic stub prior source (plan §24, defaults).

    Resolves priors from the versioned repair-family declarations (FI-034);
    may be overridden per parameter with ``overrides`` (a dict of
    ``"FAMILY.param" -> value``).  Never raises: an unknown family/parameter
    simply yields ``None`` (uniform fallback).
    """

    registry: RepairFamilyRegistry = field(default_factory=RepairFamilyRegistry.default)
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", dict(self.overrides or {}))

    def prior_for(self, family: str, parameter: str) -> Optional[Any]:
        key = f"{family}.{parameter}"
        if key in self.overrides:
            return self.overrides[key]
        try:
            declaration = self.registry.get(family)
        except ValueError:
            return None
        return declaration.parameter_prior.value_of(parameter)


def _sample_param(
    param: ConditionalParameter,
    rng: random.Random,
    provider: Optional[ConditionalPriorProvider],
    family: str,
) -> Any:
    """Draw a parameter value, honoring the injectable prior with a bias.

    The prior provider (plan §24) anchors the draw: with a fixed probability
    the prior value is proposed verbatim (cheap exploitation of domain
    knowledge); otherwise the parameter samples its conditional domain.  A
    ``None`` prior falls through to a uniform draw — never a dense grid.
    """
    prior = None
    if provider is not None:
        prior = provider.prior_for(family, param.name)
    if prior is None:
        prior = param.prior  # declared static prior
    if prior is not None:
        # Validate the injected prior against the conditional domain.
        try:
            tmp = ConditionalParameter(
                name=param.name,
                kind=param.kind,
                choices=param.choices,
                low=param.low,
                high=param.high,
                prior=prior,
            )
            prior = tmp.prior  # normalized (float coerced)
        except ValueError:
            prior = None  # injected prior outside domain → uniform fallback
    if prior is not None and rng.random() < 0.5:
        return prior
    return param.sample(rng)


# ---------------------------------------------------------------------------
# Hierarchical conditional search strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalRecord:
    """A recorded conditional observation (family + params + utility).

    Attributes:
        family: Upper-snake repair family name.
        params: Conditional parameter values for the family.
        utility: Recorded utility (direction-signed, higher = better).
    """

    family: str
    params: Tuple[Tuple[str, Any], ...]
    utility: float

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "params": list(self.params), "utility": float(self.utility)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRecord":
        return cls(
            family=data["family"],
            params=tuple((k, v) for k, v in data["params"]),
            utility=float(data["utility"]),
        )


class HierarchicalConditionalSearch(SearchStrategy):
    """TPE-style hierarchical conditional search (plan §24).

    Learns ``p(family, parameters | good)`` over a conditional tree:

    - warmup: quasi-random coverage of every family (uniform over its
      conditional parameter domain);
    - conditional TPE: for every (family, parameter) pair, maintain the
      good/bad utility split of recorded *conditional* observations (rows are
      family-specific, so a CAUSAL_SMOOTHING good row never contaminates a
      U_SHAPE_REPAIR model);
    - proposal: score each family by its good/bad density ratio (l/g), then
      sample the winner's parameters by family-specific 1-D KDE over the
      good observations of *that family's* parameters (with a prior-anchored
      uniform fallback before enough history).

    The proposal carries ``{"family": <str>, "params": {...}}`` verbatim — no
    value is coerced to a float index (the categorical identity stays intact).

    Parameters:
        tree: Ordered conditional families (``build_conditional_tree``).
        n_initial: Warmup proposals per family before TPE takes over.
        gamma: Fraction of recorded observations treated as "good".
        n_candidates: TPE candidates per proposal.
        seed: RNG seed.
        prior_provider: Optional injectable prior source (plan §24).
    """

    def __init__(
        self,
        tree: Sequence[ConditionalFamily],
        n_initial: int = 3,
        gamma: float = 0.25,
        n_candidates: int = 24,
        seed: Optional[int] = None,
        prior_provider: Optional[ConditionalPriorProvider] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ) -> None:
        if not isinstance(tree, (list, tuple)) or not tree:
            raise ValueError("hierarchical conditional search requires a non-empty tree")
        if not all(isinstance(f, ConditionalFamily) for f in tree):
            raise TypeError("tree entries must be ConditionalFamily instances")
        family_names = [f.family for f in tree]
        if len(family_names) != len(set(family_names)):
            raise ValueError("conditional tree family names must be unique")
        if not isinstance(n_initial, int) or isinstance(n_initial, bool) or n_initial < 0:
            raise ValueError("n_initial must be a non-negative integer")
        if not isinstance(gamma, numbers.Real) or isinstance(gamma, bool) or not (0 < gamma <= 1):
            raise ValueError("gamma must be a number in (0, 1]")
        if not isinstance(n_candidates, int) or isinstance(n_candidates, bool) or n_candidates < 1:
            raise ValueError("n_candidates must be a positive integer")
        # Base strategy with a placeholder numeric space (the conditional
        # grammar is handled natively; never routed to numeric kernels).
        from factor_optimizer.search.strategies import ParameterSpace, SearchSpace
        super().__init__(
            SearchSpace(parameters=[ParameterSpace("__probe__", "float", 0.0, 1.0)]),
            seed=seed,
            objective_spec=objective_spec,
            context=context,
        )
        self._tree = list(tree)
        self._tree_map = {f.family: f for f in tree}
        self.n_initial = n_initial
        self.gamma = gamma
        self.n_candidates = n_candidates
        self._prior_provider = prior_provider
        self._records: List[ConditionalRecord] = []
        self._random = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # warmup: shuffled queue covering every family once, then cycling.
        self._warmup_queue: List[str] = []
        self._rebuild_warmup_queue()

    # -- helpers --------------------------------------------------------------

    def _rebuild_warmup_queue(self) -> None:
        queue = list(self._tree_map.keys())
        self._random.shuffle(queue)
        self._warmup_queue = queue

    def _draw_params(self, family: ConditionalFamily) -> Dict[str, Any]:
        return {
            p.name: _sample_param(p, self._random, self._prior_provider, family.family)
            for p in family.parameters
        }

    def _family_counts(self, rows: Sequence[ConditionalRecord]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row.family] = counts.get(row.family, 0) + 1
        return counts

    def _kde_sample(self, samples: Sequence[float], low: float, high: float, n: int) -> np.ndarray:
        if not samples:
            return np.array([self._random.uniform(low, high) for _ in range(n)])
        vals = np.asarray(samples, dtype=np.float64)
        std = max(float(np.std(vals)), 1e-8)
        bandwidth = max(1.06 * std * (len(vals) ** -0.2), 1e-8)
        centres = vals
        chosen_idx = self._random.choices(range(len(centres)), k=n)
        chosen = centres[chosen_idx]
        noise = self.np_rng.normal(0, bandwidth, size=n)
        return np.clip(chosen + noise, low, high)

    def _kde_score(self, x: float, samples: Sequence[float], low: float, high: float) -> float:
        if not samples:
            return 1.0 / max(high - low, 1e-12)
        vals = np.asarray(samples, dtype=np.float64)
        std = max(float(np.std(vals)), 1e-8)
        bandwidth = max(1.06 * std * (len(vals) ** -0.2), 1e-8)
        diff = (x - vals) / bandwidth
        density = np.exp(-0.5 * diff ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        return float(np.mean(density))

    def _sample_conditional_params(
        self, family: ConditionalFamily, good_rows: Sequence[ConditionalRecord]
    ) -> Dict[str, Any]:
        """Sample one candidate's params conditional on a *family-specific* good split."""
        good = [row for row in good_rows if row.family == family.family]
        candidate: Dict[str, Any] = {}
        for param in family.parameters:
            if param.kind == "choice":
                # Discrete: empirical good frequency (laplace-smoothed) rather
                # than a numeric KDE — a choice value has no numeric distance.
                good_choices: Dict[Any, int] = {}
                for row in good:
                    value = dict(row.params).get(param.name)
                    if value is not None:
                        good_choices[value] = good_choices.get(value, 0) + 1
                eps = 1e-6
                weights = [good_choices.get(c, 0) + eps for c in param.choices]
                total = sum(weights)
                probs = [w / total for w in weights]
                r = self._random.random()
                acc = 0.0
                for choice, prob in zip(param.choices, probs):
                    acc += prob
                    if r <= acc:
                        candidate[param.name] = choice
                        break
                else:  # pragma: no cover - floating accumulation edge
                    candidate[param.name] = param.choices[-1]
            else:
                low, high = param.low, param.high
                good_vals = [float(dict(row.params)[param.name]) for row in good
                             if param.name in dict(row.params)]
                x = float(self._kde_sample(good_vals, low, high, 1)[0])
                candidate[param.name] = int(round(x)) if param.kind == "int" else x
        return candidate

    def _propose_dict(self) -> Dict[str, Any]:
        if len(self._records) < self.n_initial * len(self._tree):
            if not self._warmup_queue:
                self._rebuild_warmup_queue()
            name = self._warmup_queue.pop()
            family = self._tree_map[name]
            return {"family": family.family, "params": self._draw_params(family)}

        sorted_records = sorted(self._records, key=lambda r: r.utility, reverse=True)
        n_good = max(1, int(math.ceil(self.gamma * len(sorted_records))))
        good = sorted_records[:n_good]
        bad = sorted_records[n_good:]
        good_counts = self._family_counts(good)
        bad_counts = self._family_counts(bad)
        eps = 1e-6

        best_family_name: Optional[str] = None
        best_ratio = -math.inf
        for family in self._tree:
            l = good_counts.get(family.family, 0) + eps
            g = bad_counts.get(family.family, 0) + eps
            ratio = l / g
            if ratio > best_ratio:
                best_ratio = ratio
                best_family_name = family.family
        assert best_family_name is not None
        family = self._tree_map[best_family_name]

        best_score = -math.inf
        best_params: Dict[str, Any] = {}
        for _ in range(self.n_candidates):
            params = self._sample_conditional_params(family, good)
            # Score by the family's conditional l/g (parameters drawn from the
            # good distribution); no dense grid is ever built.
            log_l = 0.0
            log_g = 0.0
            good_rows_family = [r for r in good if r.family == family.family]
            bad_rows_family = [r for r in bad if r.family == family.family]
            for param in family.parameters:
                value = params[param.name]
                if param.kind == "choice":
                    l_prob = sum(1.0 for r in good_rows_family
                                 if dict(r.params).get(param.name) == value) + eps
                    l_prob /= (len(good_rows_family) + eps * len(param.choices))
                    g_prob = sum(1.0 for r in bad_rows_family
                                 if dict(r.params).get(param.name) == value) + eps
                    g_prob /= (len(bad_rows_family) + eps * len(param.choices))
                else:
                    low, high = param.low, param.high
                    good_vals = [float(dict(r.params)[param.name]) for r in good_rows_family
                                 if param.name in dict(r.params)]
                    bad_vals = [float(dict(r.params)[param.name]) for r in bad_rows_family
                                if param.name in dict(r.params)]
                    l_prob = self._kde_score(float(value), good_vals, low, high)
                    g_prob = self._kde_score(float(value), bad_vals, low, high)
                log_l += math.log(max(l_prob, 1e-300))
                log_g += math.log(max(g_prob, 1e-300))
            score = log_l - log_g
            if score > best_score:
                best_score = score
                best_params = params
        return {"family": family.family, "params": best_params}

    def propose_recipe(self) -> Dict[str, Any]:
        """Return the next conditional candidate as a plain dict."""
        return self._propose_dict()

    # -- SearchStrategy interface --------------------------------------------

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        cand = self._propose_dict()
        params = dict(cand["params"])
        params["family"] = cand["family"]
        self._next_proposal_index()
        return Trial(
            trial_id=trial_id or self._make_trial_id("conditional", params),
            mutation_id="conditional-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "hierarchical-conditional"},
        )

    def record(self, params: Dict[str, Any], score: float) -> None:
        super().record(params, score)
        family = params.get("family") or params.get("recipe")
        if family is None:
            return
        utility = self._context.to_utility(float(score))
        cond_params = {k: v for k, v in params.items() if k not in ("family", "recipe")}
        self._records.append(
            ConditionalRecord(
                family=str(family),
                params=tuple(sorted(cond_params.items())),
                utility=utility,
            )
        )

    # -- checkpoint / resume --------------------------------------------------

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "tree": [f.to_dict() for f in self._tree],
                "n_initial": self.n_initial,
                "gamma": self.gamma,
                "n_candidates": self.n_candidates,
                "seed": self.seed,
                "context": self._context.to_dict(),
            },
            "records": [r.to_dict() for r in self._records],
            "warmup_queue": list(self._warmup_queue),
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        self._records = [ConditionalRecord.from_dict(r) for r in fields.get("records", [])]
        self._warmup_queue = list(fields.get("warmup_queue", []))
        self.np_rng = np.random.default_rng(self.seed)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self.save_state()))

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "HierarchicalConditionalSearch":
        raw = dict(_from_jsonable(dict(data)))
        fields = raw.get("strategy_fields", {})
        ctor = fields.get("ctor", {})
        tree = tuple(ConditionalFamily.from_dict(f) for f in ctor.get("tree", []))
        kwargs: Dict[str, Any] = {
            "tree": tree,
            "n_initial": ctor.get("n_initial", 3),
            "gamma": ctor.get("gamma", 0.25),
            "n_candidates": ctor.get("n_candidates", 24),
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
