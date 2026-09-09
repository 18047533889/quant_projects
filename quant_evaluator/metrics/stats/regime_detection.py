"""
Hidden Markov Model (HMM) regime detection for time series.

Regime-switching models capture periods of distinct market behavior (bull/bear,
high/low volatility). HMMs estimate latent states and transition probabilities
using the Baum-Welch (EM) algorithm.

References:
    Baum, L.E.; Petrie, T. (1966). "Statistical Inference for Probabilistic
    Functions of Finite State Markov Chains". Annals of Mathematical Statistics.
    37 (6): 1554-1563. doi:10.1214/aoms/1177699147

    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle". Econometrica. 57 (2):
    357-384. doi:10.2307/1912559

    Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected
    Applications in Speech Recognition". Proceedings of the IEEE. 77 (2):
    257-286. doi:10.1109/5.18626
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.special import logsumexp

@dataclass(frozen=True)
class RegimeInference:
    probabilities: np.ndarray
    states: np.ndarray
    scope: str
    fit_end: Optional[object]
    decision_time: Optional[object]
    production_eligible: bool

    def __post_init__(self):
        if self.scope not in {"FILTERED_ASOF", "SMOOTHED_POSTHOC"}:
            raise ValueError("invalid inference scope")
        p, s = np.asarray(self.probabilities), np.asarray(self.states)
        if p.ndim != 2 or s.shape != (len(p),) or not np.isfinite(p).all():
            raise ValueError("invalid posterior/state axes")
        if self.fit_end is None or self.decision_time is None:
            raise ValueError("typed inference requires fit_end and decision_time")
        try:
            ordered = self.decision_time > self.fit_end
        except TypeError as exc:
            raise ValueError("fit_end and decision_time must share an ordered clock") from exc
        if not ordered:
            raise ValueError("decision_time must be strictly after fit_end")
        if self.production_eligible != (self.scope == "FILTERED_ASOF"):
            raise ValueError("production eligibility contradicts inference scope")

def require_production_inference(result: RegimeInference) -> RegimeInference:
    if not isinstance(result, RegimeInference) or not result.production_eligible or result.scope != "FILTERED_ASOF":
        raise ValueError("production preprocessing requires FILTERED_ASOF regime inference")
    return result


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.

    Each state emits observations from a Gaussian distribution with state-specific
    mean and variance.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        missing_policy: str = "reject",
    ):
        """
        Initialize HMM.

        Args:
            n_states: Number of hidden states
            n_iter: Maximum iterations for EM algorithm
            tol: Convergence tolerance for log-likelihood
            random_state: Random seed for initialization
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        if missing_policy not in {"reject"}:
            raise ValueError("only explicit missing_policy='reject' is currently supported")
        self.missing_policy = missing_policy
        if isinstance(n_iter, bool) or not isinstance(n_iter, (int,np.integer)) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        if isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite")

        # Model parameters (fitted during training)
        self.start_prob_ = None  # (n_states,) initial state distribution
        self.trans_mat_ = None  # (n_states, n_states) transition matrix
        self.means_ = None  # (n_states,) emission means
        self.vars_ = None  # (n_states,) emission variances

        self.converged_ = False
        self.n_iter_fit_ = 0
        self.log_likelihood_history_ = []

    def _initialize_params(self, X: np.ndarray) -> None:
        """
        Initialize HMM parameters.

        Args:
            X: Observations (T,)
        """
        rng = np.random.RandomState(self.random_state)

        # Start probability: uniform
        self.start_prob_ = np.ones(self.n_states) / self.n_states

        # Transition matrix: slight diagonal preference + noise
        self.trans_mat_ = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.trans_mat_, 0.7)
        self.trans_mat_ /= self.trans_mat_.sum(axis=1, keepdims=True)

        # Emission parameters: split data into quantiles
        quantiles = np.linspace(0, 1, self.n_states + 1)
        self.means_ = np.zeros(self.n_states)
        self.vars_ = np.zeros(self.n_states)

        for i in range(self.n_states):
            q_low = np.quantile(X, quantiles[i])
            q_high = np.quantile(X, quantiles[i + 1])
            mask = (X >= q_low) & (X <= q_high)
            if np.sum(mask) > 0:
                self.means_[i] = np.mean(X[mask])
                self.vars_[i] = np.var(X[mask]) + 1e-6
            else:
                self.means_[i] = np.mean(X)
                self.vars_[i] = np.var(X) + 1e-6

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities.

        Args:
            X: Observations (T,)

        Returns:
            Log probabilities (T, n_states)
        """
        T = len(X)
        log_prob = np.zeros((T, self.n_states))

        for state in range(self.n_states):
            mean = self.means_[state]
            var = self.vars_[state]
            # Log of Gaussian PDF
            log_prob[:, state] = stats.norm.logpdf(X, loc=mean, scale=np.sqrt(var))

        return log_prob

    def _forward(self, log_emission: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm for computing state probabilities.

        Args:
            log_emission: Log emission probabilities (T, n_states)

        Returns:
            (log_alpha, log_likelihood)
            log_alpha: Forward log-probabilities (T, n_states)
            log_likelihood: Total log-likelihood of sequence
        """
        T = log_emission.shape[0]
        log_alpha = np.zeros((T, self.n_states))

        # Initialize: α_0(s) = π(s) * b_s(x_0)
        log_alpha[0] = np.log(self.start_prob_ + 1e-10) + log_emission[0]

        # Recursion: α_t(s) = b_s(x_t) * Σ_s' α_{t-1}(s') * A_{s',s}
        log_trans = np.log(self.trans_mat_ + 1e-10)

        for t in range(1, T):
            for s in range(self.n_states):
                log_alpha[t, s] = logsumexp(log_alpha[t - 1] + log_trans[:, s]) + log_emission[t, s]

        # Total likelihood: Σ_s α_T(s)
        log_likelihood = logsumexp(log_alpha[-1])

        return log_alpha, log_likelihood

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        """
        Backward algorithm.

        Args:
            log_emission: Log emission probabilities (T, n_states)

        Returns:
            log_beta: Backward log-probabilities (T, n_states)
        """
        T = log_emission.shape[0]
        log_beta = np.zeros((T, self.n_states))

        # Initialize: β_T(s) = 1 (log = 0)
        # Already initialized to 0

        # Recursion: β_t(s) = Σ_s' A_{s,s'} * b_{s'}(x_{t+1}) * β_{t+1}(s')
        log_trans = np.log(self.trans_mat_ + 1e-10)

        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                log_beta[t, s] = logsumexp(
                    log_trans[s, :] + log_emission[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _compute_posteriors(
        self, log_alpha: np.ndarray, log_beta: np.ndarray, log_emission: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior state probabilities (E-step).

        Args:
            log_alpha: Forward log-probabilities (T, n_states)
            log_beta: Backward log-probabilities (T, n_states)

        Returns:
            (gamma, xi)
            gamma: Marginal state probabilities (T, n_states)
            xi: Pairwise state probabilities (T-1, n_states, n_states)
        """
        T = log_alpha.shape[0]

        # γ_t(s) = P(S_t = s | X) = α_t(s) * β_t(s) / P(X)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # ξ_t(i,j) = P(S_t=i, S_{t+1}=j | X)
        #          = α_t(i) * A_{i,j} * b_j(x_{t+1}) * β_{t+1}(j) / P(X)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        log_trans = np.log(self.trans_mat_ + 1e-10)

        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + log_trans[i, j]
                        + log_emission[t + 1, j]
                        + log_beta[t + 1, j]
                    )

            # Normalize
            xi[t] -= logsumexp(xi[t])
            xi[t] = np.exp(xi[t])

        return gamma, xi

    def _update_params(self, gamma: np.ndarray, xi: np.ndarray, X: np.ndarray) -> None:
        """
        Update model parameters (M-step).

        Args:
            gamma: Marginal state probabilities (T, n_states)
            xi: Pairwise state probabilities (T-1, n_states, n_states)
            X: Observations (T,)
        """
        T = len(X)

        # Update start probabilities: π(s) = γ_0(s)
        self.start_prob_ = gamma[0]

        # Update transition matrix: A_{i,j} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        xi_sum = np.sum(xi, axis=0)  # (n_states, n_states)
        gamma_sum = np.sum(gamma[:-1], axis=0)  # (n_states,)
        self.trans_mat_ = xi_sum / (gamma_sum[:, np.newaxis] + 1e-10)

        # Normalize rows
        self.trans_mat_ /= self.trans_mat_.sum(axis=1, keepdims=True)

        # Update emission parameters
        for s in range(self.n_states):
            # Weighted mean: Σ_t γ_t(s) * x_t / Σ_t γ_t(s)
            weights = gamma[:, s]
            weight_sum = np.sum(weights)

            if weight_sum > 1e-10:
                self.means_[s] = np.sum(weights * X) / weight_sum
                self.vars_[s] = np.sum(weights * (X - self.means_[s]) ** 2) / weight_sum
                # Ensure non-zero variance
                self.vars_[s] = max(self.vars_[s], 1e-6)

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            X: Observations (T,) or (T, 1)

        Returns:
            self
        """
        # Invalidate first: any failed refit must make the old fitted model unusable.
        self.start_prob_=self.trans_mat_=self.means_=self.vars_=None
        self.converged_=False; self.n_iter_fit_=0; self.log_likelihood_history_=[]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2 and X.shape[1] == 1: X=X[:,0]
        elif X.ndim != 1: raise ValueError(f"X must have shape (T,) or (T,1), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("NaN/Inf requires a declared event-clock missing transition model; missing_policy='reject'")

        T = len(X)
        if T < 2 * self.n_states:
            raise ValueError(f"Insufficient data: T={T} < {2*self.n_states}")

        # Initialize parameters
        self._initialize_params(X)

        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        # EM iterations
        for iteration in range(self.n_iter):
            # E-step
            log_emission = self._compute_log_likelihood(X)
            log_alpha, log_likelihood = self._forward(log_emission)
            log_beta = self._backward(log_emission)
            gamma, xi = self._compute_posteriors(log_alpha, log_beta, log_emission)

            # M-step
            self._update_params(gamma, xi, X)
            if (not np.isfinite(self.trans_mat_).all() or
                    not np.allclose(self.trans_mat_.sum(axis=1), 1.0)):
                self.start_prob_=self.trans_mat_=self.means_=self.vars_=None
                self.n_iter_fit_=0
                raise FloatingPointError("invalid transition matrix after M-step")

            self.log_likelihood_history_.append(log_likelihood)

            # Check convergence
            if iteration > 0:
                delta = log_likelihood - prev_log_likelihood
                if delta < -max(self.tol, 1e-8):
                    self.start_prob_=self.trans_mat_=self.means_=self.vars_=None
                    self.n_iter_fit_=0
                    raise FloatingPointError(f"EM likelihood decreased by {delta}")
                if 0 <= delta < self.tol:
                    self.converged_ = True
                    self.n_iter_fit_ = iteration + 1
                    break

            prev_log_likelihood = log_likelihood

        if not self.converged_:
            self.n_iter_fit_ = self.n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: Observations (T,)

        Returns:
            State sequence (T,) with values in [0, n_states-1]
        """
        if self.n_iter_fit_ == 0 or self.means_ is None: raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2 and X.shape[1] == 1: X=X[:,0]
        elif X.ndim != 1: raise ValueError("X must have shape (T,) or (T,1)")
        if not np.isfinite(X).all(): raise ValueError("missing observations rejected by event-clock contract")
        T = len(X)

        log_emission = self._compute_log_likelihood(X)
        log_trans = np.log(self.trans_mat_ + 1e-10)
        log_start = np.log(self.start_prob_ + 1e-10)

        # Viterbi: δ_t(s) = max probability path ending in state s at time t
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=np.int32)

        # Initialize
        log_delta[0] = log_start + log_emission[0]

        # Recursion
        for t in range(1, T):
            for s in range(self.n_states):
                candidates = log_delta[t - 1] + log_trans[:, s]
                psi[t, s] = np.argmax(candidates)
                log_delta[t, s] = candidates[psi[t, s]] + log_emission[t, s]

        # Backtrack
        states = np.zeros(T, dtype=np.int32)
        states[-1] = np.argmax(log_delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray, scope="smoothed_posthoc") -> np.ndarray:
        """
        Compute posterior state probabilities.

        Args:
            X: Observations (T,)

        Returns:
            State probabilities (T, n_states)
        """
        canonical={"smoothed_posthoc":"SMOOTHED_POSTHOC","filtered_asof":"FILTERED_ASOF",
                   "SMOOTHED_POSTHOC":"SMOOTHED_POSTHOC","FILTERED_ASOF":"FILTERED_ASOF"}.get(scope)
        if canonical is None: raise ValueError("unknown inference scope")
        return self._scope_probabilities(X, canonical)

    def _scope_probabilities(self, X, scope):
        if self.n_iter_fit_ == 0 or self.means_ is None: raise RuntimeError("model is not fitted")
        X=np.asarray(X,dtype=float)
        if X.ndim==2 and X.shape[1]==1: X=X[:,0]
        elif X.ndim!=1: raise ValueError("X must have shape (T,) or (T,1)")
        if not np.isfinite(X).all(): raise ValueError("missing observations rejected by event-clock contract")
        emission=self._compute_log_likelihood(X); alpha,_=self._forward(emission)
        if scope=="FILTERED_ASOF": gamma=alpha
        elif scope=="SMOOTHED_POSTHOC": gamma=alpha+self._backward(emission)
        else: raise ValueError("scope must be FILTERED_ASOF or SMOOTHED_POSTHOC")
        gamma-=logsumexp(gamma,axis=1,keepdims=True)
        return np.exp(gamma)

    def infer(self, X, *, scope="FILTERED_ASOF", fit_end=None, decision_time=None):
        """Typed inference; only filtered probabilities are production eligible."""
        probabilities=self._scope_probabilities(X,scope)
        states=np.argmax(probabilities,axis=1).astype(np.int32)
        return RegimeInference(probabilities,states,scope,fit_end,decision_time,scope=="FILTERED_ASOF")


def detect_regimes(
    data: np.ndarray,
    n_states: int = 2,
    n_iter: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, GaussianHMM]:
    """
    Detect market regimes using Hidden Markov Model.

    Args:
        data: Time series (T,)
        n_states: Number of regimes (default 2 for bull/bear)
        n_iter: Maximum EM iterations
        random_state: Random seed

    Returns:
        (states, probabilities, model)
        states: Most likely regime sequence (T,)
        probabilities: Posterior regime probabilities (T, n_states)
        model: Fitted GaussianHMM object

    Example:
        >>> returns = np.random.randn(1000) * 0.01
        >>> states, probs, model = detect_regimes(returns, n_states=2)
        >>> print(f"Regime means: {model.means_}")
        >>> print(f"Regime volatilities: {np.sqrt(model.vars_)}")
    """
    model = GaussianHMM(n_states=n_states, n_iter=n_iter, random_state=random_state)
    model.fit(data)

    states = model.predict(data)
    probabilities = model.predict_proba(data)

    return states, probabilities, model
