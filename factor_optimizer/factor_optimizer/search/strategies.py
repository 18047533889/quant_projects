"""Search strategies: Random, Grid, Bayesian (GP), and TPE."""

import hashlib
import itertools
import json
import math
import numbers
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.trial import Trial, TrialStatus


# ---------------------------------------------------------------------------
# Serializable search-strategy state
# ---------------------------------------------------------------------------

@dataclass
class SearchStrategyState:
    """Snapshot of a search strategy for checkpoint/resume.

    Carries the full stochastic state of a strategy so a resumed search
    reproduces the EXACT same proposal trajectory as an uninterrupted one:

    - ``rng_state``: Python ``random.Random`` state (``random.getstate()``).
    - ``np_rng_state``: NumPy ``Generator`` state (``rng.bit_generator.state``).
    - ``history``: recorded ``(params, score)`` evaluations.
    - ``iteration_count``: number of proposals already emitted.
    - ``strategy_fields``: strategy-specific attributes (index, grid, kernel
      state, ...).
    - ``extra_rng_states``: auxiliary RNG states indexed by name (e.g. the
      ``RandomSearch`` companion inside Bayesian/TPE search).

    All values are JSON-safe (plain dicts/lists/numbers) so the state can be
    embedded in a ``SearchSession`` checkpoint or written standalone.
    """

    strategy_type: str
    rng_state: Dict[str, Any]
    np_rng_state: Optional[Dict[str, Any]] = None
    history: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    iteration_count: int = 0
    strategy_fields: Dict[str, Any] = field(default_factory=dict)
    extra_rng_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyContext:
    """Single direction authority handed to a strategy by the runner.

    The runner derives this from the authoritative ``ObjectiveSpec`` so a
    strategy can never disagree with the session's objective.  Internally the
    strategy maximizes ``utility = direction_sign * raw_metric`` where
    ``direction_sign`` is ``+1`` for ``maximize`` and ``-1`` for ``minimize``.
    """

    objective_spec: ObjectiveSpec

    @property
    def direction(self) -> str:
        return self.objective_spec.direction

    @property
    def direction_sign(self) -> int:
        return 1 if self.objective_spec.direction == "maximize" else -1

    def to_utility(self, raw_metric: float) -> float:
        """Map a raw metric to the internal utility convention (maximize)."""
        return self.direction_sign * float(raw_metric)

    def to_dict(self) -> Dict[str, Any]:
        return {"objective_spec": self.objective_spec.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyContext":
        if not isinstance(data, dict) or "objective_spec" not in data:
            raise ValueError("StrategyContext checkpoint missing 'objective_spec'")
        return cls(objective_spec=ObjectiveSpec.from_dict(data["objective_spec"]))


def _canonical_params(params: Dict[str, Any]) -> str:
    """Stable, order-independent canonical serialization of a parameter dict.

    Used as input to the deterministic trial identity so that two proposals
    with the same parameter values hash identically regardless of dict
    insertion order.
    """
    return json.dumps(
        {k: params[k] for k in sorted(params)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _jsonable(value: Any) -> Any:
    """Convert a value into plain JSON-safe structures.

    Tuples are tagged as ``{"__tuple__": [...]}`` so tuple-typed state (Python
    ``random`` state vectors, ``(name, X, y)`` rows, ...) survives a JSON
    round-trip with its exact type intact.
    """
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _from_jsonable(value: Any) -> Any:
    """Inverse of ``_jsonable``."""
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_from_jsonable(item) for item in value["__tuple__"])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _capture_python_rng(rng: random.Random) -> Dict[str, Any]:
    """Serialize a Python ``random.Random`` state to a JSON-safe dict."""
    return {"state": _jsonable(rng.getstate())}


def _restore_python_rng(rng: random.Random, payload: Dict[str, Any]) -> None:
    """Restore a Python ``random.Random`` state captured by ``_capture_python_rng``."""
    rng.setstate(_from_jsonable(payload["state"]))


def _capture_np_rng(rng: np.random.Generator) -> Dict[str, Any]:
    """Serialize a NumPy ``Generator`` state to a JSON-safe dict."""
    return _jsonable(rng.bit_generator.state)


def _restore_np_rng(rng: np.random.Generator, payload: Dict[str, Any]) -> None:
    """Restore a NumPy ``Generator`` state captured by ``_capture_np_rng``.

    ``rng.bit_generator.state`` is a plain dict, so the JSON round-trip is
    lossless for PCG64 (the generator created by ``np.random.default_rng``).
    The state is assigned through ``bit_generator.state`` (equivalent to
    ``__setstate__`` but uniformly honored across numpy versions).
    """
    rng.bit_generator.state = dict(payload)


def _capture_np_global_state() -> Dict[str, Any]:
    """Serialize the global NumPy MT19937 state (``np.random.*``)."""
    keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "keys": [int(k) for k in keys],
        "pos": int(pos),
        "has_gauss": bool(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_np_global_state(payload: Dict[str, Any]) -> None:
    """Restore the global NumPy MT19937 state captured by ``_capture_np_global_state``."""
    np.random.set_state(
        (
            "MT19937",
            np.array(payload["keys"], dtype=np.uint32),
            payload["pos"],
            payload["has_gauss"],
            payload["cached_gaussian"],
        )
    )


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

@dataclass
class ParameterSpace:
    """Describes the search space for a single parameter.

    A parameter can be:
    - ``"float"``: continuous ``[low, high]``
    - ``"int"``: integer ``[low, high]``
    - ``"choice"``: categorical from a list of values

    ``log_scale`` applies to ``float`` and ``int`` parameters: bounds are
    interpreted in log-space and an integer parameter will be rounded to the
    nearest integer within the bounds.
    """

    name: str
    kind: str  # "float", "int", "choice"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate the parameter definition fail-closed (FO-P0-07).

        An invalid parameter (unknown kind, non-numeric or inverted bounds,
        empty categorical choices) is rejected at construction instead of
        producing NaN/None proposals or silently wrong results downstream.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        if self.kind not in ("float", "int", "choice"):
            raise ValueError(
                f"unknown parameter kind: {self.kind!r} (must be "
                f"'float', 'int', or 'choice')"
            )
        if self.kind in ("float", "int"):
            if not isinstance(self.low, numbers.Real) or isinstance(
                self.low, bool
            ):
                raise ValueError(
                    f"{self.kind} parameter '{self.name}' requires a numeric "
                    f"'low' bound"
                )
            if not isinstance(self.high, numbers.Real) or isinstance(
                self.high, bool
            ):
                raise ValueError(
                    f"{self.kind} parameter '{self.name}' requires a numeric "
                    f"'high' bound"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"{self.kind} parameter '{self.name}' requires low < high, "
                    f"got low={self.low!r}, high={self.high!r}"
                )
            if self.kind == "int":
                # FO-P1-03: an int parameter's bounds must be integral.  A
                # fractional bound would silently truncate/round in sampling
                # and produce a range that disagrees with the declared bounds.
                if float(self.low) != int(self.low) or float(self.high) != int(self.high):
                    raise ValueError(
                        f"int parameter '{self.name}' requires integral low/high "
                        f"bounds, got low={self.low!r}, high={self.high!r}"
                    )
            if self.log_scale:
                if self.low <= 0:
                    raise ValueError(
                        f"log_scale {self.kind} parameter '{self.name}' "
                        f"requires low > 0, got low={self.low!r}"
                    )
                if self.high <= 0:
                    raise ValueError(
                        f"log_scale {self.kind} parameter '{self.name}' "
                        f"requires high > 0, got high={self.high!r}"
                    )
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(
                    f"choice parameter '{self.name}' must have a non-empty "
                    f"'choices' list"
                )
            if not all(isinstance(c, str) for c in self.choices):
                raise ValueError(
                    f"choice parameter '{self.name}' requires string choices; "
                    f"numeric kernels cannot consume non-string categorical "
                    f"values"
                )
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(
                    f"choice parameter '{self.name}' requires unique choices"
                )
            if self.log_scale:
                raise ValueError(
                    f"log_scale is not applicable to choice parameter "
                    f"'{self.name}'"
                )

    def validate_for_numeric_kernel(self) -> None:
        """Reject parameters a numeric GP/TPE kernel cannot represent.

        BayesianSearch's surrogate operates on a float64 vector of every
        parameter.  A categorical ``choice`` parameter has no numeric
        distance/sampling semantics, so passing one to the Gaussian/TPE
        numeric handling is a fail-closed ``ValueError`` (FO-P0-03) rather
        than a garbage float proposal.
        """
        if self.kind == "choice":
            raise ValueError(
                f"categorical parameter '{self.name}' is not supported by "
                f"BayesianSearch/TPESearch numeric kernels; restrict the "
                f"search space to 'float' and 'int' parameters for these "
                f"strategies"
            )

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        """Draw a uniform random sample from this parameter's domain."""
        rng = rng or random
        if self.kind == "float":
            if self.log_scale:
                log_low = math.log(float(self.low))
                log_high = math.log(float(self.high))
                return math.exp(rng.uniform(log_low, log_high))
            return rng.uniform(self.low, self.high)
        elif self.kind == "int":
            if self.log_scale:
                log_low = math.log(float(self.low))
                log_high = math.log(float(self.high))
                val = math.exp(rng.uniform(log_low, log_high))
                return int(max(self.low, min(self.high, round(val))))
            return rng.randint(int(self.low), int(self.high))
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(f"choice parameter '{self.name}' has no choices")
            return rng.choice(self.choices)
        else:
            raise ValueError(f"unknown parameter kind: {self.kind}")

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize this parameter definition to a JSON-safe dict."""
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "log_scale": self.log_scale,
        }

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "ParameterSpace":
        """Deserialize a parameter definition captured by ``to_checkpoint_dict``."""
        return cls(**dict(data))

    def to_grid(self, n_points: int = 10) -> List[Any]:
        """Return a list of evenly-spaced values for grid enumeration."""
        if self.kind == "float":
            if self.log_scale:
                return list(np.geomspace(float(self.low), float(self.high), n_points))
            return list(np.linspace(self.low, self.high, n_points))
        elif self.kind == "int":
            if self.log_scale:
                vals = np.geomspace(float(self.low), float(self.high), n_points)
                return sorted(set(int(round(v)) for v in vals))
            return list(range(int(self.low), int(self.high) + 1))
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(f"choice parameter '{self.name}' has no choices")
            return list(self.choices)
        else:
            raise ValueError(f"unknown parameter kind: {self.kind}")

    def bounds(self) -> Tuple[float, float]:
        """Return (low, high) as floats for numeric parameters."""
        if self.kind in ("float", "int"):
            return float(self.low), float(self.high)
        raise ValueError(f"bounds() not applicable to choice parameter '{self.name}'")


@dataclass
class SearchSpace:
    """Ordered collection of parameter definitions."""

    parameters: List[ParameterSpace]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, list) or not all(
            isinstance(p, ParameterSpace) for p in self.parameters
        ):
            raise ValueError(
                "SearchSpace parameters must be a list of ParameterSpace"
            )
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(
                f"SearchSpace parameter names must be unique, got {names}"
            )

    def require_numeric_kernel(self, strategy_name: str) -> None:
        """Fail closed if any parameter lacks numeric kernel semantics.

        Raises ``ValueError`` (naming *strategy_name*) when a ``choice``
        parameter would otherwise be fed to a Gaussian-process / TPE numeric
        kernel (FO-P0-03).
        """
        for p in self.parameters:
            if p.kind == "choice":
                raise ValueError(
                    f"categorical parameter '{p.name}' is not supported by "
                    f"{strategy_name}; use only 'float'/'int' parameters for "
                    f"numeric-kernel strategies"
                )

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Draw a random point from the full space."""
        return {p.name: p.sample(rng) for p in self.parameters}

    def grid(self, n_points: int = 10) -> List[Dict[str, Any]]:
        """Enumerate the Cartesian product of per-parameter grids."""
        per_param = [p.to_grid(n_points) for p in self.parameters]
        keys = [p.name for p in self.parameters]
        return [dict(zip(keys, combo)) for combo in itertools.product(*per_param)]

    def validate_point(self, point: Dict[str, Any]) -> None:
        """Validate a proposed point against this space, fail-closed (FO-P1-03).

        Checks, for every parameter:
        - the exact key set matches the space (no missing, no extra keys);
        - the value type matches the parameter kind;
        - numeric values are finite and within ``[low, high]``;
        - ``int`` values are integral;
        - ``log_scale`` values are positive;
        - ``choice`` values are one of the declared choices.

        Raises ``ValueError`` on the first violation.
        """
        if not isinstance(point, dict):
            raise ValueError("point must be a dict of parameter values")
        expected = {p.name for p in self.parameters}
        actual = set(point)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            detail = []
            if missing:
                detail.append(f"missing keys: {missing}")
            if extra:
                detail.append(f"unexpected keys: {extra}")
            raise ValueError("point keys do not match search space: " + "; ".join(detail))
        for p in self.parameters:
            value = point[p.name]
            if p.kind == "float":
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise ValueError(
                        f"parameter '{p.name}' must be a real number, got {type(value).__name__}"
                    )
                fval = float(value)
                if not math.isfinite(fval):
                    raise ValueError(f"parameter '{p.name}' must be finite")
                if fval < float(p.low) or fval > float(p.high):
                    raise ValueError(
                        f"parameter '{p.name}' value {fval!r} out of bounds "
                        f"[{p.low}, {p.high}]"
                    )
                if p.log_scale and fval <= 0:
                    raise ValueError(
                        f"log_scale parameter '{p.name}' must be positive, got {fval!r}"
                    )
            elif p.kind == "int":
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise ValueError(
                        f"parameter '{p.name}' must be an integer, got {type(value).__name__}"
                    )
                if float(value) != int(value):
                    raise ValueError(
                        f"parameter '{p.name}' must be integral, got {value!r}"
                    )
                ival = int(value)
                if ival < int(p.low) or ival > int(p.high):
                    raise ValueError(
                        f"parameter '{p.name}' value {ival!r} out of bounds "
                        f"[{p.low}, {p.high}]"
                    )
                if p.log_scale and ival <= 0:
                    raise ValueError(
                        f"log_scale parameter '{p.name}' must be positive, got {ival!r}"
                    )
            elif p.kind == "choice":
                if value not in p.choices:
                    raise ValueError(
                        f"parameter '{p.name}' value {value!r} is not one of "
                        f"the declared choices {p.choices}"
                    )
            else:  # pragma: no cover - guarded at construction
                raise ValueError(f"unknown parameter kind: {p.kind}")

    def to_array(self, point: Dict[str, Any]) -> np.ndarray:
        """Convert a named parameter dict to a 1-D numpy array (float64).

        Fail-closed: a categorical value has no numeric representation and
        raises instead of producing a ``float('nan')`` / garbage entry.
        """
        if any(p.kind == "choice" for p in self.parameters):
            raise ValueError(
                "to_array requires a numeric-only search space; categorical "
                "parameters have no float representation"
            )
        return np.array([float(point[p.name]) for p in self.parameters], dtype=np.float64)

    def from_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Convert a 1-D numpy array back to a named parameter dict."""
        result: Dict[str, Any] = {}
        for i, p in enumerate(self.parameters):
            val = arr[i]
            if p.kind == "int":
                val = int(round(val))
                val = max(p.low, min(p.high, val))
            elif p.kind == "choice":
                idx = int(round(val))
                idx = max(0, min(len(p.choices) - 1, idx))
                val = p.choices[idx]
            result[p.name] = val
        return result

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize this search space to a JSON-safe dict."""
        return {"parameters": [p.to_checkpoint_dict() for p in self.parameters]}

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        """Deserialize a search space captured by ``to_checkpoint_dict``."""
        if not isinstance(data, dict) or "parameters" not in data:
            raise ValueError("search space checkpoint missing 'parameters'")
        return cls(
            parameters=[
                ParameterSpace.from_checkpoint_dict(payload)
                for payload in data["parameters"]
            ]
        )


# ---------------------------------------------------------------------------
# Base strategy interface
# ---------------------------------------------------------------------------

class SearchStrategy(ABC):
    """Base class for all search strategies.

    Subclasses must implement ``proposal_fn`` which is compatible with
    ``SearchRunner``'s callback pattern: a callable that returns a ``Trial``
    with suggested parameters stored in ``trial.metadata["params"]``.

    A ``SearchStrategy`` may be constructed with a space containing
    categorical ``choice`` parameters ONLY if its proposal logic supports
    them (``RandomSearch`` and ``GridSearch`` do).  Strategies whose kernels
    require numeric inputs (``BayesianSearch``, ``TPESearch``) reject
    categorical parameters fail-closed at construction (FO-P0-03).
    """

    def __init__(
        self,
        space: SearchSpace,
        seed: Optional[int] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ):
        if not isinstance(space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        self.space = space
        self.seed = seed
        self.rng = random.Random(seed)
        self._history: List[Tuple[Dict[str, Any], float]] = []
        # FO-P1-02: ONE central proposal counter owned by the base strategy.
        # Every emitted proposal increments it exactly once; per-strategy
        # ``_iteration`` drift is removed.  Checkpoint/resume restores it.
        self._proposal_sequence: int = 0
        # FO-P0-03: single direction authority.  The strategy's direction is
        # derived from the ObjectiveSpec (via StrategyContext), never from an
        # independent per-strategy ``maximize`` flag.  When constructed
        # directly (backward-compatible), the default spec is maximize.
        if context is not None:
            if not isinstance(context, StrategyContext):
                raise TypeError("context must be a StrategyContext (or None)")
            self._context = context
        else:
            spec = objective_spec if objective_spec is not None else ObjectiveSpec.from_direction("maximize")
            if not isinstance(spec, ObjectiveSpec):
                raise TypeError("objective_spec must be an ObjectiveSpec (or None)")
            self._context = StrategyContext(objective_spec=spec)
        self._direction_sign = self._context.direction_sign

    @property
    def objective_spec(self) -> ObjectiveSpec:
        """The authoritative objective spec driving this strategy's direction."""
        return self._context.objective_spec

    @property
    def direction(self) -> str:
        """The strategy's objective direction, always derived from the spec."""
        return self._context.direction

    @property
    def proposal_sequence(self) -> int:
        """Number of proposals emitted so far (single central counter)."""
        return self._proposal_sequence

    def _next_proposal_index(self) -> int:
        """Increment the central counter and return the new proposal index.

        Every emitted proposal calls this exactly once, so the counter is the
        single source of truth for proposal ordering across all strategies.
        """
        self._proposal_sequence += 1
        return self._proposal_sequence

    def _make_trial_id(
        self,
        strategy_id: str,
        params: Dict[str, Any],
        parent_lineage: Optional[Sequence[str]] = None,
    ) -> str:
        """Derive a deterministic scientific trial identity (FO-P1-01).

        The identity is a sha256 over (search_session_content_id,
        strategy_id, proposal_index, canonical params, parent lineage) so the
        same proposal in the same session always yields the same trial_id —
        no uuid4 nondeterminism.  Operational UUIDs (retries, executions) are
        kept separate as ``execution_attempt_id``.
        """
        session_cid = getattr(self, "_search_session_content_id", None) or ""
        lineage = list(parent_lineage) if parent_lineage else []
        payload = "|".join(
            [
                str(session_cid),
                strategy_id,
                str(self._proposal_sequence),
                _canonical_params(params),
                "|".join(str(item) for item in lineage),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _check_categorical_support(self, strategy_name: str) -> None:
        """Fail closed if this space holds categorical parameters this
        strategy cannot handle.

        Strategies whose kernels require numeric inputs (``BayesianSearch``,
        ``TPESearch``) call this at construction; any categorical parameter
        raises (FO-P0-03, FO-P0-07).  ``RandomSearch`` and ``GridSearch``
        support categorical parameters natively and do not call it.
        """
        for p in self.space.parameters:
            if p.kind == "choice":
                raise ValueError(
                    f"categorical parameter '{p.name}' is not supported by "
                    f"{strategy_name}; use only 'float'/'int' parameters for "
                    f"{strategy_name}"
                )

    def record(self, params: Dict[str, Any], score: float) -> None:
        """Record a completed evaluation for strategies that learn from history."""
        self._history.append((params, score))

    @abstractmethod
    def propose(self, trial_id: Optional[str] = None) -> Trial:
        """Suggest a new trial with parameters in ``metadata["params"]``."""
        ...

    def proposal_fn(self) -> Trial:
        """Callable compatible with ``SearchRunner(proposal_fn=...)``.

        Default implementation delegates to ``self.propose()``.
        """
        return self.propose()

    # -- checkpoint/resume ----------------------------------------------------

    def save_state(self) -> SearchStrategyState:
        """Capture this strategy's full stochastic state.

        Includes the Python RNG state, NumPy RNG state (where used), recorded
        history, iteration count, and strategy-specific fields.  A strategy
        restored from this state reproduces the exact same proposal trajectory
        as one that never stopped.
        """
        return SearchStrategyState(
            strategy_type=type(self).__name__,
            rng_state=_capture_python_rng(self.rng),
            np_rng_state=(
                _capture_np_rng(self.np_rng)
                if getattr(self, "np_rng", None) is not None
                else None
            ),
            history=[(dict(params), score) for params, score in self._history],
            iteration_count=self._proposal_sequence,
            strategy_fields=self._state_fields(),
            extra_rng_states=self._extra_rng_states(),
        )

    def load_state(self, state: SearchStrategyState) -> None:
        """Restore this strategy's full stochastic state from ``state``.

        Raises ``ValueError`` if ``state`` was captured by a different strategy
        type.
        """
        if state.strategy_type != type(self).__name__:
            raise ValueError(
                f"cannot restore {type(self).__name__} from a checkpoint "
                f"captured by {state.strategy_type}"
            )
        _restore_python_rng(self.rng, state.rng_state)
        if getattr(self, "np_rng", None) is not None:
            if state.np_rng_state is None:
                raise ValueError(
                    f"{type(self).__name__} checkpoint is missing NumPy RNG state"
                )
            _restore_np_rng(self.np_rng, state.np_rng_state)
        self._history = [(dict(params), float(score)) for params, score in state.history]
        if getattr(self, "_X", None) is not None:
            self._X = [np.asarray(row, dtype=np.float64) for row in self._X]
        self._proposal_sequence = int(state.iteration_count)
        self._restore_state_fields(state.strategy_fields)
        self._restore_extra_rng_states(state.extra_rng_states)

    # -- extension points (defaults are no-ops) --------------------------------

    def _state_fields(self) -> Dict[str, Any]:
        """Strategy-specific JSON-safe state to checkpoint (overridden)."""
        return {}

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        """Restore strategy-specific fields (overridden)."""

    def _extra_rng_states(self) -> Dict[str, Dict[str, Any]]:
        """Auxiliary RNG states by name (overridden)."""
        return {}

    def _restore_extra_rng_states(self, states: Dict[str, Dict[str, Any]]) -> None:
        """Restore auxiliary RNG states by name (overridden)."""

    # -- shared JSON persistence ----------------------------------------------

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize this strategy to a JSON-safe dict (standalone checkpoint)."""
        return _jsonable(asdict(self.save_state()))

    def save_checkpoint(self, path: str) -> None:
        """Write this strategy's state to a JSON file at *path*."""
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_checkpoint_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "SearchStrategy":
        """Deserialize a standalone strategy checkpoint.

        ``data`` must carry a ``strategy_type`` and a ``strategy_fields`` key
        describing the strategy's constructor arguments.  Restores the strategy
        and its full stochastic state in one call.
        """
        raw = dict(_from_jsonable(dict(data)))
        strategy_type = raw.get("strategy_type")
        fields = raw.get("strategy_fields")
        if not isinstance(strategy_type, str) or not strategy_type.strip():
            raise ValueError("strategy checkpoint missing a strategy_type")
        if not isinstance(fields, dict) or "ctor" not in fields:
            raise ValueError("strategy checkpoint missing ctor fields")
        ctor = fields["ctor"]
        if not isinstance(ctor, dict):
            raise ValueError("strategy checkpoint ctor fields must be a dict")
        kind = strategy_type.lower()
        if "random" in kind:
            strategy = cls._from_ctor_kwargs(RandomSearch, ctor)
        elif "grid" in kind:
            strategy = cls._from_ctor_kwargs(GridSearch, ctor)
        elif "bayesian" in kind:
            strategy = cls._from_ctor_kwargs(BayesianSearch, ctor)
        elif "tpe" in kind:
            strategy = cls._from_ctor_kwargs(TPESearch, ctor)
        else:
            raise ValueError(f"unsupported strategy_type: {strategy_type}")
        strategy.load_state(SearchStrategyState(**raw))
        return strategy

    @staticmethod
    def _from_ctor_kwargs(
        strategy_cls: type, ctor: Dict[str, Any]
    ) -> "SearchStrategy":
        if "space" not in ctor:
            raise ValueError("strategy checkpoint ctor fields must include 'space'")
        space = SearchSpace.from_checkpoint_dict(ctor["space"])
        kwargs = dict(ctor)
        kwargs["space"] = space
        # Restore the single direction authority from the checkpoint so a
        # resumed strategy keeps the exact same objective direction.
        if "context" in kwargs:
            kwargs["context"] = StrategyContext.from_dict(kwargs["context"])
        return strategy_cls(**kwargs)

    @classmethod
    def load_checkpoint(cls, path: str) -> "SearchStrategy":
        """Load and restore a standalone strategy checkpoint from *path*."""
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_checkpoint_dict(data)


# ---------------------------------------------------------------------------
# 1. Random Search
# ---------------------------------------------------------------------------

class RandomSearch(SearchStrategy):
    """Uniform random sampling over the parameter space.

    RandomSearch supports every ``ParameterSpace`` kind, including
    categorical ``choice`` parameters (sampled uniformly).
    """

    def __init__(
        self,
        space: SearchSpace,
        seed: Optional[int] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ):
        super().__init__(space, seed, objective_spec=objective_spec, context=context)
        # RandomSearch natively supports categorical choice parameters.

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        params = self.space.sample(self.rng)
        self._next_proposal_index()
        return Trial(
            trial_id=trial_id or self._make_trial_id("random", params),
            mutation_id="random-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "random"},
        )

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "space": self.space.to_checkpoint_dict(),
                "seed": self.seed,
                "context": self._context.to_dict(),
            }
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# 2. Grid Search
# ---------------------------------------------------------------------------

class GridSearch(SearchStrategy):
    """Grid enumeration with optional lazy grid reduction (FO-P1-04).

    Parameters:
        space: Search space definition.
        grid_points: Number of grid points per numeric parameter (default 10).
        max_combinations: If set, sample this many points from the full grid
            rather than enumerating everything.
        seed: RNG seed for reproducibility.

    GridSearch supports categorical ``choice`` parameters by enumerating each
    choice as a grid value.

    The full Cartesian product is NEVER materialized.  The grid is described
    by its per-parameter cardinalities and a mixed-radix index space; proposals
    are produced by decoding a combination index into concrete parameter
    values.  When ``max_combinations`` is set, a deterministic subset of
    combination indices is sampled (seeded) and decoded lazily, so a grid with
    10^8 combinations and ``max_combinations=1000`` allocates only 1000
    points, not the full product.
    """

    def __init__(
        self,
        space: SearchSpace,
        grid_points: int = 10,
        max_combinations: Optional[int] = None,
        seed: Optional[int] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ):
        super().__init__(space, seed, objective_spec=objective_spec, context=context)
        self.grid_points = grid_points
        self._max_combinations = max_combinations
        if not isinstance(grid_points, int) or isinstance(grid_points, bool) or grid_points < 1:
            raise ValueError("grid_points must be a positive integer")
        if max_combinations is not None and (
            not isinstance(max_combinations, int)
            or isinstance(max_combinations, bool)
            or max_combinations < 1
        ):
            raise ValueError("max_combinations must be a positive integer or None")
        # Per-parameter grid values (small: grid_points per numeric param, or
        # the choice list).  The Cartesian product of these is NOT built.
        self._per_param_values: List[List[Any]] = [
            p.to_grid(grid_points) for p in self.space.parameters
        ]
        self._keys = [p.name for p in self.space.parameters]
        # Mixed-radix cardinalities: the number of values per axis.
        self._radices = [len(vals) for vals in self._per_param_values]
        # Total number of combinations (the theoretical grid size).
        self._total_combinations = 1
        for radix in self._radices:
            self._total_combinations *= radix
        # Deterministic subset of combination indices to visit.
        if max_combinations is not None and max_combinations < self._total_combinations:
            self._indices = self.rng.sample(
                range(self._total_combinations), max_combinations
            )
        else:
            self._indices = list(range(self._total_combinations))
        self._index = 0

    @property
    def total_combinations(self) -> int:
        """The theoretical grid size (never materialized)."""
        return self._total_combinations

    def _decode(self, combo_index: int) -> Dict[str, Any]:
        """Decode a mixed-radix combination index into concrete parameters."""
        params: Dict[str, Any] = {}
        remaining = combo_index
        for key, radix, values in zip(self._keys, self._radices, self._per_param_values):
            axis = remaining % radix
            remaining //= radix
            params[key] = values[axis]
        return params

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        if self._index >= len(self._indices):
            raise StopIteration("grid exhausted")
        combo_index = self._indices[self._index]
        self._index += 1
        self._next_proposal_index()
        params = self._decode(combo_index)
        return Trial(
            trial_id=trial_id or self._make_trial_id("grid", params),
            mutation_id="grid-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "grid"},
        )

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "space": self.space.to_checkpoint_dict(),
                "grid_points": self.grid_points,
                "max_combinations": self._max_combinations,
                "seed": self.seed,
                "context": self._context.to_dict(),
            },
            "index": self._index,
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        if "index" in fields:
            self._index = int(fields["index"])


# ---------------------------------------------------------------------------
# 3. Bayesian Optimization (GP surrogate)
# ---------------------------------------------------------------------------

class BayesianSearch(SearchStrategy):
    """Bayesian optimization with a Gaussian Process surrogate.

    Uses a simple square-exponential (RBF) kernel.  Acquisition function
    is Expected Improvement (EI) by default, with Upper Confidence Bound
    (UCB) as an alternative.

    Parameters:
        space: Search space definition.
        n_initial: Number of random initial points before the GP kicks in.
        acquisition: ``"ei"`` or ``"ucb"``.
        ucb_kappa: Exploration weight for UCB (ignored for EI).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        space: SearchSpace,
        n_initial: int = 5,
        acquisition: str = "ei",
        ucb_kappa: float = 2.576,
        seed: Optional[int] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ):
        super().__init__(space, seed, objective_spec=objective_spec, context=context)
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.ucb_kappa = ucb_kappa
        if not isinstance(n_initial, int) or isinstance(n_initial, bool) or n_initial < 0:
            raise ValueError("n_initial must be a non-negative integer")
        if acquisition not in ("ei", "ucb"):
            raise ValueError("acquisition must be 'ei' or 'ucb'")
        if not isinstance(ucb_kappa, numbers.Real) or isinstance(ucb_kappa, bool):
            raise ValueError("ucb_kappa must be a finite number")
        if not math.isfinite(ucb_kappa):
            raise ValueError("ucb_kappa must be a finite number")
        # FO-P0-03 / FO-P0-07: the GP surrogate operates on a numeric vector;
        # a categorical parameter cannot be represented and would produce a
        # garbage float proposal.  Fail closed at construction.
        self._check_categorical_support("BayesianSearch")
        # The companion RandomSearch shares the same single direction authority.
        self._random = RandomSearch(space, seed, context=self._context)
        self._X: List[np.ndarray] = []
        # FO-P0-03: ``_y`` stores UTILITY (direction_sign * raw_metric), so the
        # GP always maximizes utility regardless of the objective direction.
        self._y: List[float] = []
        self._length_scale = 1.0
        self._noise = 1e-6

    def record(self, params: Dict[str, Any], score: float) -> None:
        super().record(params, score)
        self._X.append(self.space.to_array(params))
        # Store utility so the surrogate's "best" is always the max utility.
        self._y.append(self._context.to_utility(score))
        self._update_kernel()

    def _update_kernel(self) -> None:
        """Heuristic length-scale update: median heuristic on observed X."""
        if len(self._X) < 2:
            return
        X = np.array(self._X)
        # Pairwise distances
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        # Median heuristic (skip zero diagonal)
        off_diag = dists[np.triu_indices(len(dists), k=1)]
        if len(off_diag) > 0 and np.median(off_diag) > 0:
            self._length_scale = float(np.median(off_diag))

    def _gp_predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at *X_new* using the training set."""
        if not self._X:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        X_train = np.array(self._X)
        y_train = np.array(self._y)
        n = len(X_train)
        ls = self._length_scale
        noise = self._noise

        def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            diffs = a[:, np.newaxis, :] - b[np.newaxis, :, :]
            sq = np.sum(diffs ** 2, axis=-1)
            return np.exp(-0.5 * sq / (ls ** 2))

        K = rbf(X_train, X_train) + noise * np.eye(n)
        K_s = rbf(X_train, X_new)
        K_ss = rbf(X_new, X_new)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        var = np.clip(var, 1e-12, None)
        return mu, var

    def _expected_improvement(
        self, mu: np.ndarray, var: np.ndarray, best_y: float
    ) -> np.ndarray:
        """Compute Expected Improvement acquisition."""
        sigma = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_y) / sigma
            ei = (mu - best_y) * _norm_cdf(z) + sigma * _norm_pdf(z)
            ei[sigma < 1e-12] = 0.0
        return ei

    def _ucb(self, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Compute Upper Confidence Bound acquisition."""
        return mu + self.ucb_kappa * np.sqrt(var)

    def _optimize_acquisition(self, n_candidates: int = 5000) -> np.ndarray:
        """Sample candidates and pick the one with highest acquisition."""
        best_acq = -np.inf
        best_x = np.zeros(self.space.dim)

        for _ in range(n_candidates):
            x_cand = np.array([self.space.parameters[i].sample(self.rng)
                               for i in range(self.space.dim)])
            X_cand = x_cand.reshape(1, -1)
            mu, var = self._gp_predict(X_cand)
            if self.acquisition == "ei" and self._y:
                acq = self._expected_improvement(mu, var, max(self._y))[0]
            elif self.acquisition == "ucb":
                acq = self._ucb(mu, var)[0]
            else:
                # Fallback: random
                return x_cand
            if acq > best_acq:
                best_acq = acq
                best_x = x_cand
        return best_x

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        # Not enough data yet -- random initialization.  Sample from the
        # companion RandomSearch's RNG (so its state is checkpointed) but build
        # the Trial here so the central counter increments exactly once and the
        # trial identity is derived from this strategy.
        if len(self._X) < self.n_initial:
            params = self.space.sample(self._random.rng)
            self._next_proposal_index()
            return Trial(
                trial_id=trial_id or self._make_trial_id("bayesian", params),
                mutation_id="bayesian-search",
                status=TrialStatus.PROPOSED,
                metadata={"params": params, "strategy": "bayesian"},
            )

        best_x = self._optimize_acquisition()
        self._next_proposal_index()
        params = self.space.from_array(best_x)
        return Trial(
            trial_id=trial_id or self._make_trial_id("bayesian", params),
            mutation_id="bayesian-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "bayesian"},
        )

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "space": self.space.to_checkpoint_dict(),
                "n_initial": self.n_initial,
                "acquisition": self.acquisition,
                "ucb_kappa": self.ucb_kappa,
                "seed": self.seed,
                "context": self._context.to_dict(),
            },
            "X": [row.tolist() for row in self._X],
            "y": list(self._y),
            "length_scale": self._length_scale,
            "noise": self._noise,
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        self._X = [np.asarray(row, dtype=np.float64) for row in fields.get("X", [])]
        self._y = [float(value) for value in fields.get("y", [])]
        self._length_scale = float(fields.get("length_scale", 1.0))
        self._noise = float(fields.get("noise", 1e-6))
        self._X = [np.asarray(row, dtype=np.float64) for row in self._X]

    def _extra_rng_states(self) -> Dict[str, Dict[str, Any]]:
        return {"_random": _capture_python_rng(self._random.rng)}

    def _restore_extra_rng_states(self, states: Dict[str, Dict[str, Any]]) -> None:
        if "_random" in states:
            _restore_python_rng(self._random.rng, states["_random"])


# ---------------------------------------------------------------------------
# 4. Tree-structured Parzen Estimator (TPE)
# ---------------------------------------------------------------------------

class TPESearch(SearchStrategy):
    """Tree-structured Parzen Estimator (TPE).

    Sorts observations by score, fits separate KDEs for the "good" and "bad"
    partitions, and samples from the good distribution weighted by the
    likelihood ratio l(x)/g(x).

    Parameters:
        space: Search space definition.
        n_initial: Number of random initial points.
        gamma: Fraction of observations treated as "good" (best gamma * n).
        n_candidates: Number of candidate points sampled per proposal.
        seed: RNG seed for reproducibility.
        maximize: Backward-compatible convenience.  If True, higher scores are
            "good"; if False, lower scores are "good".  Defaults to False
            (minimize) for historical compatibility.  When ``objective_spec``
            or ``context`` is also supplied, ``maximize`` must agree with it;
            the spec/context is the single direction authority (FO-P0-03).
        objective_spec: The authoritative ObjectiveSpec (single direction
            authority).  When supplied, the strategy's direction is derived
            from it and ``maximize`` cannot disagree.
        context: A StrategyContext carrying the authoritative ObjectiveSpec.
    """

    def __init__(
        self,
        space: SearchSpace,
        n_initial: int = 5,
        gamma: float = 0.25,
        n_candidates: int = 24,
        seed: Optional[int] = None,
        maximize: Optional[bool] = None,
        objective_spec: Optional[ObjectiveSpec] = None,
        context: Optional[StrategyContext] = None,
    ):
        if not isinstance(n_initial, int) or isinstance(n_initial, bool) or n_initial < 0:
            raise ValueError("n_initial must be a non-negative integer")
        if not isinstance(gamma, numbers.Real) or isinstance(gamma, bool):
            raise ValueError("gamma must be a number in (0, 1]")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be a number in (0, 1]")
        if not isinstance(n_candidates, int) or isinstance(n_candidates, bool) or n_candidates < 1:
            raise ValueError("n_candidates must be a positive integer")
        if maximize is not None and not isinstance(maximize, bool):
            raise ValueError("maximize must be a boolean")
        # Resolve the single direction authority.  The spec/context wins; the
        # legacy ``maximize`` flag is reconciled against it and rejected if it
        # disagrees, so the strategy can never drift from the objective.
        if context is not None:
            if not isinstance(context, StrategyContext):
                raise TypeError("context must be a StrategyContext (or None)")
            eff_context = context
        elif objective_spec is not None:
            if not isinstance(objective_spec, ObjectiveSpec):
                raise TypeError("objective_spec must be an ObjectiveSpec (or None)")
            eff_context = StrategyContext(objective_spec=objective_spec)
        else:
            # Backward-compatible default: the ObjectiveSpec defaults to
            # maximize (matching the runner's default), unless the legacy
            # ``maximize`` flag is explicitly supplied.
            if maximize is None:
                direction = "maximize"
            else:
                direction = "maximize" if maximize else "minimize"
            eff_context = StrategyContext(
                objective_spec=ObjectiveSpec.from_direction(direction)
            )
        if maximize is not None:
            spec_dir = eff_context.direction
            if (maximize and spec_dir != "maximize") or (
                not maximize and spec_dir != "minimize"
            ):
                raise ValueError(
                    "maximize and objective_spec/context disagree; the "
                    "objective spec is the single direction authority"
                )
        super().__init__(space, seed, context=eff_context)
        self.n_initial = n_initial
        self.gamma = gamma
        self.n_candidates = n_candidates
        # FO-P0-03 / FO-P0-07: TPE's Parzen-estimator kernels require numeric
        # parameters.  Reject categorical parameters fail-closed at
        # construction rather than producing garbage proposals.
        self._check_categorical_support("TPESearch")
        # Dedicated NumPy RNG so all stochastic ops are reproducible from seed.
        self.np_rng = np.random.default_rng(seed)
        self._random = RandomSearch(space, seed, context=self._context)

    def record(self, params: Dict[str, Any], score: float) -> None:
        # Store UTILITY (direction_sign * raw_metric) so the "good" partition
        # is always the highest-utility observations regardless of direction.
        self._history.append((dict(params), self._context.to_utility(score)))

    def _kde_sample(
        self,
        samples: List[Dict[str, Any]],
        low: float,
        high: float,
        n: int,
    ) -> np.ndarray:
        """Sample from a 1-D Gaussian KDE fitted to *samples*."""
        if not samples:
            return np.array([self.rng.uniform(low, high) for _ in range(n)])
        vals = np.array(samples, dtype=np.float64)
        std = max(np.std(vals), 1e-8)
        bandwidth = 1.06 * std * (len(vals) ** -0.2)
        bandwidth = max(bandwidth, 1e-8)
        centres = vals
        chosen_idx = self.rng.choices(range(len(centres)), k=n)
        chosen = centres[chosen_idx]
        noise = self.np_rng.normal(0, bandwidth, size=n)
        result = chosen + noise
        result = np.clip(result, low, high)
        return result

    def _kde_score(
        self,
        x: float,
        samples: List[float],
        low: float,
        high: float,
    ) -> float:
        """Evaluate the Gaussian KDE density at point *x*."""
        if not samples:
            # Uniform fallback
            return 1.0 / max(high - low, 1e-12)
        vals = np.array(samples, dtype=np.float64)
        std = max(np.std(vals), 1e-8)
        bandwidth = 1.06 * std * (len(vals) ** -0.2)
        bandwidth = max(bandwidth, 1e-8)
        diff = (x - vals) / bandwidth
        density = np.exp(-0.5 * diff ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        return float(np.mean(density))

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        if len(self._history) < self.n_initial:
            params = self.space.sample(self._random.rng)
            self._next_proposal_index()
            return Trial(
                trial_id=trial_id or self._make_trial_id("tpe", params),
                mutation_id="tpe-search",
                status=TrialStatus.PROPOSED,
                metadata={"params": params, "strategy": "tpe"},
            )

        # History stores UTILITY (direction_sign * raw_metric), so the "good"
        # group is always the highest-utility observations — the strategy
        # always maximizes utility regardless of the objective direction.
        sorted_hist = sorted(self._history, key=lambda t: t[1], reverse=True)
        n_good = max(1, int(math.ceil(self.gamma * len(sorted_hist))))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:]

        good_params = [p for p, _ in good]
        bad_params = [p for p, _ in bad]

        best_score = -np.inf
        best_params: Dict[str, Any] = {}

        for _ in range(self.n_candidates):
            candidate: Dict[str, Any] = {}
            log_l = 0.0
            log_g = 0.0

            for param_def in self.space.parameters:
                name = param_def.name

                if param_def.kind == "float":
                    low, high = param_def.bounds()
                    good_vals = [p[name] for p in good_params]
                    bad_vals = [p[name] for p in bad_params]
                    x = float(self._kde_sample(good_vals, low, high, 1)[0])
                    candidate[name] = x
                    l_prob = self._kde_score(x, good_vals, low, high)
                    g_prob = self._kde_score(x, bad_vals, low, high)
                    log_l += math.log(max(l_prob, 1e-300))
                    log_g += math.log(max(g_prob, 1e-300))

                elif param_def.kind == "int":
                    low, high = param_def.bounds()
                    good_vals = [float(p[name]) for p in good_params]
                    bad_vals = [float(p[name]) for p in bad_params]
                    x = float(self._kde_sample(good_vals, low, high, 1)[0])
                    candidate[name] = int(round(x))
                    l_prob = self._kde_score(x, good_vals, low, high)
                    g_prob = self._kde_score(x, bad_vals, low, high)
                    log_l += math.log(max(l_prob, 1e-300))
                    log_g += math.log(max(g_prob, 1e-300))

                elif param_def.kind == "choice":
                    # Unreachable in normal operation: TPESearch rejects
                    # categorical parameters at construction (FO-P0-03).
                    raise ValueError(
                        f"categorical parameter '{name}' is not supported by "
                        f"TPESearch"
                    )

            # Score: l(x) / g(x) -- higher is better
            score = log_l - log_g
            if score > best_score:
                best_score = score
                best_params = candidate

        self._next_proposal_index()
        return Trial(
            trial_id=trial_id or self._make_trial_id("tpe", best_params),
            mutation_id="tpe-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": best_params, "strategy": "tpe"},
        )

    def _state_fields(self) -> Dict[str, Any]:
        return {
            "ctor": {
                "space": self.space.to_checkpoint_dict(),
                "n_initial": self.n_initial,
                "gamma": self.gamma,
                "n_candidates": self.n_candidates,
                "seed": self.seed,
                "context": self._context.to_dict(),
            },
        }

    def _restore_state_fields(self, fields: Dict[str, Any]) -> None:
        pass

    def _extra_rng_states(self) -> Dict[str, Dict[str, Any]]:
        return {"_random": _capture_python_rng(self._random.rng)}

    def _restore_extra_rng_states(self, states: Dict[str, Dict[str, Any]]) -> None:
        if "_random" in states:
            _restore_python_rng(self._random.rng, states["_random"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF (no scipy dependency)."""
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
