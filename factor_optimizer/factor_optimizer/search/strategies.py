"""Search strategies: Random, Grid, Bayesian (GP), and TPE."""

import itertools
import math
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from factor_optimizer.contracts.trial import Trial, TrialStatus


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

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        """Draw a uniform random sample from this parameter's domain."""
        rng = rng or random
        if self.kind == "float":
            if self.log_scale:
                log_low = math.log(max(self.low, 1e-300))
                log_high = math.log(max(self.high, 1e-300))
                return math.exp(rng.uniform(log_low, log_high))
            return rng.uniform(self.low, self.high)
        elif self.kind == "int":
            if self.log_scale:
                log_low = math.log(max(self.low, 1))
                log_high = math.log(max(self.high, 1))
                val = math.exp(rng.uniform(log_low, log_high))
                return int(max(self.low, min(self.high, round(val))))
            return rng.randint(self.low, self.high)
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(f"choice parameter '{self.name}' has no choices")
            return rng.choice(self.choices)
        else:
            raise ValueError(f"unknown parameter kind: {self.kind}")

    def to_grid(self, n_points: int = 10) -> List[Any]:
        """Return a list of evenly-spaced values for grid enumeration."""
        if self.kind == "float":
            if self.log_scale:
                return list(np.geomspace(max(self.low, 1e-300), max(self.high, 1e-300), n_points))
            return list(np.linspace(self.low, self.high, n_points))
        elif self.kind == "int":
            if self.log_scale:
                vals = np.geomspace(max(self.low, 1), max(self.high, 1), n_points)
                return sorted(set(int(round(v)) for v in vals))
            return list(range(self.low, self.high + 1))
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

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Draw a random point from the full space."""
        return {p.name: p.sample(rng) for p in self.parameters}

    def grid(self, n_points: int = 10) -> List[Dict[str, Any]]:
        """Enumerate the Cartesian product of per-parameter grids."""
        per_param = [p.to_grid(n_points) for p in self.parameters]
        keys = [p.name for p in self.parameters]
        return [dict(zip(keys, combo)) for combo in itertools.product(*per_param)]

    def to_array(self, point: Dict[str, Any]) -> np.ndarray:
        """Convert a named parameter dict to a 1-D numpy array (float64)."""
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


# ---------------------------------------------------------------------------
# Base strategy interface
# ---------------------------------------------------------------------------

class SearchStrategy(ABC):
    """Base class for all search strategies.

    Subclasses must implement ``proposal_fn`` which is compatible with
    ``SearchRunner``'s callback pattern: a callable that returns a ``Trial``
    with suggested parameters stored in ``trial.metadata["params"]``.
    """

    def __init__(self, space: SearchSpace, seed: Optional[int] = None):
        self.space = space
        self.seed = seed
        self.rng = random.Random(seed)
        self._history: List[Tuple[Dict[str, Any], float]] = []

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


# ---------------------------------------------------------------------------
# 1. Random Search
# ---------------------------------------------------------------------------

class RandomSearch(SearchStrategy):
    """Uniform random sampling over the parameter space."""

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        params = self.space.sample(self.rng)
        return Trial(
            trial_id=trial_id or f"random-{uuid.uuid4().hex[:12]}",
            mutation_id="random-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "random"},
        )


# ---------------------------------------------------------------------------
# 2. Grid Search
# ---------------------------------------------------------------------------

class GridSearch(SearchStrategy):
    """Grid enumeration with optional grid reduction.

    Parameters:
        space: Search space definition.
        grid_points: Number of grid points per numeric parameter (default 10).
        max_combinations: If set, randomly sample this many points from the
            full grid rather than enumerating everything.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        space: SearchSpace,
        grid_points: int = 10,
        max_combinations: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(space, seed)
        self.grid_points = grid_points
        full_grid = self.space.grid(grid_points)
        if max_combinations is not None and max_combinations < len(full_grid):
            self._grid = self.rng.sample(full_grid, max_combinations)
        else:
            self._grid = full_grid
        self._index = 0

    def propose(self, trial_id: Optional[str] = None) -> Trial:
        if self._index >= len(self._grid):
            raise StopIteration("grid exhausted")
        params = self._grid[self._index]
        self._index += 1
        return Trial(
            trial_id=trial_id or f"grid-{uuid.uuid4().hex[:12]}",
            mutation_id="grid-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "grid"},
        )


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
    ):
        super().__init__(space, seed)
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.ucb_kappa = ucb_kappa
        self._random = RandomSearch(space, seed)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._length_scale = 1.0
        self._noise = 1e-6

    def record(self, params: Dict[str, Any], score: float) -> None:
        super().record(params, score)
        self._X.append(self.space.to_array(params))
        self._y.append(score)
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
        # Not enough data yet -- random initialization
        if len(self._X) < self.n_initial:
            return self._random.propose(trial_id)

        best_x = self._optimize_acquisition()
        params = self.space.from_array(best_x)
        return Trial(
            trial_id=trial_id or f"bayes-{uuid.uuid4().hex[:12]}",
            mutation_id="bayesian-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": params, "strategy": "bayesian"},
        )


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
        maximize: If True, higher scores are "good" (e.g. RankIC); if False,
            lower scores are "good". Defaults to False (minimize).
    """

    def __init__(
        self,
        space: SearchSpace,
        n_initial: int = 5,
        gamma: float = 0.25,
        n_candidates: int = 24,
        seed: Optional[int] = None,
        maximize: bool = False,
    ):
        super().__init__(space, seed)
        self.n_initial = n_initial
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.maximize = maximize
        # Dedicated NumPy RNG so all stochastic ops are reproducible from seed.
        self.np_rng = np.random.default_rng(seed)
        self._random = RandomSearch(space, seed)

    def record(self, params: Dict[str, Any], score: float) -> None:
        super().record(params, score)

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
            return self._random.propose(trial_id)

        # Sort observations by score and split into "good" / "bad" partitions.
        # The "good" group is always the best-scoring ones, which depends on
        # the objective direction: highest scores when maximizing, lowest when
        # minimizing.
        sorted_hist = sorted(
            self._history,
            key=lambda t: t[1],
            reverse=self.maximize,
        )
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
                    if not param_def.choices:
                        raise ValueError(f"choice parameter '{name}' has no choices")
                    n_choices = len(param_def.choices)
                    # Count frequencies in good and bad sets
                    good_counts = [0] * n_choices
                    bad_counts = [0] * n_choices
                    for p in good_params:
                        idx = param_def.choices.index(p[name])
                        good_counts[idx] += 1
                    for p in bad_params:
                        idx = param_def.choices.index(p[name])
                        bad_counts[idx] += 1
                    # Add Laplace smoothing
                    good_smooth = [c + 1 for c in good_counts]
                    bad_smooth = [c + 1 for c in bad_counts]
                    good_total = sum(good_smooth)
                    bad_total = sum(bad_smooth)
                    good_probs = [c / good_total for c in good_smooth]
                    bad_probs = [c / bad_total for c in bad_smooth]
                    # Sample from good distribution
                    chosen_idx = self.rng.choices(range(n_choices), weights=good_probs, k=1)[0]
                    candidate[name] = param_def.choices[chosen_idx]
                    log_l += math.log(max(good_probs[chosen_idx], 1e-300))
                    log_g += math.log(max(bad_probs[chosen_idx], 1e-300))

            # Score: l(x) / g(x) -- higher is better
            score = log_l - log_g
            if score > best_score:
                best_score = score
                best_params = candidate

        return Trial(
            trial_id=trial_id or f"tpe-{uuid.uuid4().hex[:12]}",
            mutation_id="tpe-search",
            status=TrialStatus.PROPOSED,
            metadata={"params": best_params, "strategy": "tpe"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF (no scipy dependency)."""
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
