"""Plateau detection for search stopping criteria."""

from dataclasses import dataclass
from typing import List, Optional
import statistics


@dataclass
class PlateauConfig:
    """
    Configuration for plateau detection.

    Attributes:
        window_size: Number of recent scores to consider
        min_relative_improvement: Minimum relative improvement to avoid plateau
        min_absolute_improvement: Minimum absolute improvement to avoid plateau
        require_both: Whether both criteria must be met
        use_median: Use median instead of max for baseline
    """
    window_size: int = 20
    min_relative_improvement: float = 0.001
    min_absolute_improvement: Optional[float] = None
    require_both: bool = False
    use_median: bool = False

    def __post_init__(self):
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.min_relative_improvement < 0:
            raise ValueError("min_relative_improvement must be >= 0")
        if self.min_absolute_improvement is not None and self.min_absolute_improvement < 0:
            raise ValueError("min_absolute_improvement must be >= 0")


class PlateauDetector:
    """
    Detects when search progress has plateaued.

    A plateau is detected when recent improvements fall below thresholds,
    indicating diminishing returns from continued search.
    """

    def __init__(self, config: PlateauConfig):
        """
        Initialize detector.

        Args:
            config: Plateau detection configuration
        """
        self.config = config
        self.score_history: List[float] = []

    def add_score(self, score: float) -> None:
        """Add a new score to history."""
        self.score_history.append(score)

    def is_plateau(self, scores: Optional[List[float]] = None) -> bool:
        """
        Check if current window indicates a plateau.

        Args:
            scores: Optional score list (uses internal history if None)

        Returns:
            True if plateau detected
        """
        score_list = scores if scores is not None else self.score_history

        if len(score_list) < self.config.window_size:
            return False

        # Take most recent window
        window = score_list[-self.config.window_size:]

        # Compute baseline (best so far before last element)
        if self.config.use_median:
            baseline = statistics.median(window[:-1]) if len(window) > 1 else window[0]
        else:
            # Use best score in first half of window as baseline
            mid = len(window) // 2
            baseline = max(window[:mid]) if mid > 0 else window[0]

        # Current best in the window
        current_best = max(window)

        # Check relative improvement
        relative_ok = True
        if baseline != 0:
            relative_improvement = (current_best - baseline) / abs(baseline)
            relative_ok = relative_improvement >= self.config.min_relative_improvement
        else:
            relative_ok = current_best > 0

        # Check absolute improvement (only if configured)
        absolute_ok = None
        if self.config.min_absolute_improvement is not None:
            absolute_improvement = current_best - baseline
            absolute_ok = absolute_improvement >= self.config.min_absolute_improvement

        # Apply logic
        if absolute_ok is None:
            # Only relative check
            return not relative_ok
        elif self.config.require_both:
            return not (relative_ok and absolute_ok)
        else:
            return not (relative_ok or absolute_ok)

    def reset(self) -> None:
        """Clear score history."""
        self.score_history.clear()

    def plateau_duration(self) -> int:
        """
        Return number of evaluations since last significant improvement.

        Returns:
            Number of evaluations in current plateau
        """
        if len(self.score_history) < 2:
            return 0

        threshold = self.config.min_relative_improvement
        if self.config.min_absolute_improvement is not None:
            abs_threshold = self.config.min_absolute_improvement
        else:
            abs_threshold = 0.0

        best_so_far = self.score_history[0]
        plateau_count = 0

        for score in self.score_history[1:]:
            improvement = score - best_so_far
            relative_imp = improvement / abs(best_so_far) if best_so_far != 0 else float('inf')

            if relative_imp >= threshold or improvement >= abs_threshold:
                # Significant improvement
                best_so_far = score
                plateau_count = 0
            else:
                plateau_count += 1

        return plateau_count


class AdaptivePlateauDetector:
    """
    Adaptive plateau detection with dynamic thresholds.

    Adjusts thresholds based on search phase and score volatility.
    """

    def __init__(
        self,
        initial_config: PlateauConfig,
        early_phase_window: int = 50,
        volatility_window: int = 10,
    ):
        """
        Initialize adaptive detector.

        Args:
            initial_config: Initial plateau configuration
            early_phase_window: Number of evaluations considered "early phase"
            volatility_window: Window for computing score volatility
        """
        self.base_config = initial_config
        self.early_phase_window = early_phase_window
        self.volatility_window = volatility_window
        self.score_history: List[float] = []

    def add_score(self, score: float) -> None:
        """Add a new score to history."""
        self.score_history.append(score)

    def is_plateau(self) -> bool:
        """Check if current state indicates a plateau."""
        if len(self.score_history) < self.base_config.window_size:
            return False

        # Adjust threshold based on search phase
        config = self._adapt_config()

        # Use adapted config with current window
        detector = PlateauDetector(config)
        return detector.is_plateau(self.score_history)

    def _adapt_config(self) -> PlateauConfig:
        """Adapt configuration based on search state."""
        config = PlateauConfig(
            window_size=self.base_config.window_size,
            min_relative_improvement=self.base_config.min_relative_improvement,
            min_absolute_improvement=self.base_config.min_absolute_improvement,
            require_both=self.base_config.require_both,
            use_median=self.base_config.use_median,
        )

        # In early phase, be more lenient (higher threshold = easier to plateau)
        if len(self.score_history) < self.early_phase_window:
            # Early phase: require more improvement to continue
            config.min_relative_improvement *= 2.0
            return config

        # Compute volatility
        recent = self.score_history[-self.volatility_window:]
        if len(recent) >= 2:
            volatility = statistics.stdev(recent) / (statistics.mean(recent) or 1.0)

            # High volatility: be more lenient (scores are noisy)
            if volatility > 0.1:
                config.min_relative_improvement *= 0.5

        return config

    def reset(self) -> None:
        """Clear history."""
        self.score_history.clear()


class MultiObjectivePlateauDetector:
    """
    Plateau detection for multi-objective optimization.

    Detects plateau when Pareto frontier stops expanding.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_new_nondominated: int = 1,
    ):
        """
        Initialize multi-objective detector.

        Args:
            window_size: Number of recent evaluations to consider
            min_new_nondominated: Minimum new non-dominated points to avoid plateau
        """
        self.window_size = window_size
        self.min_new_nondominated = min_new_nondominated
        self.frontier_sizes: List[int] = []

    def add_frontier_size(self, size: int) -> None:
        """Record current frontier size."""
        self.frontier_sizes.append(size)

    def is_plateau(self) -> bool:
        """Check if frontier has stopped growing."""
        if len(self.frontier_sizes) < self.window_size:
            return False

        window = self.frontier_sizes[-self.window_size:]
        initial_size = window[0]
        final_size = window[-1]

        new_points = final_size - initial_size
        return new_points < self.min_new_nondominated

    def growth_rate(self) -> float:
        """
        Compute recent frontier growth rate.

        Returns:
            Average new points per evaluation over window
        """
        if len(self.frontier_sizes) < 2:
            return 0.0

        window = self.frontier_sizes[-self.window_size:]
        if len(window) < 2:
            return 0.0

        growth = window[-1] - window[0]
        return growth / len(window)

    def reset(self) -> None:
        """Clear history."""
        self.frontier_sizes.clear()
