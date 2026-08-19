"""
Backend selector for automatic backend selection based on data size and capabilities.

Automatically selects the optimal backend (numpy, numba, cupy, polars) based on:
- Available hardware/libraries
- Data shape and size
- Operation type
- Benchmark results
"""

from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time


class BackendType(Enum):
    """Available backend types."""
    NUMPY = "numpy"
    NUMBA = "numba"
    CUPY = "cupy"
    POLARS = "polars"


@dataclass
class BackendCapabilities:
    """Backend capability information."""
    backend: BackendType
    available: bool
    supports_gpu: bool = False
    supports_jit: bool = False
    supports_dataframe: bool = False
    version: Optional[str] = None
    device_info: Optional[str] = None


class BackendSelector:
    """
    Automatic backend selector with data-size-aware heuristics.

    Selects optimal backend based on:
    1. Hardware availability (GPU, CPU cores)
    2. Data size thresholds
    3. Operation characteristics
    4. Optional benchmark results

    Examples
    --------
    >>> selector = BackendSelector()
    >>> backend = selector.select_for_rolling(T=252, N=5000)
    >>> print(backend)  # "numba" if available

    >>> # With custom thresholds
    >>> selector = BackendSelector(
    ...     numba_threshold=100_000,
    ...     cupy_threshold=1_000_000
    ... )
    """

    def __init__(
        self,
        numba_threshold: int = 50_000,
        cupy_threshold: int = 500_000,
        polars_threshold: int = 100_000,
        auto_benchmark: bool = False,
    ):
        """
        Initialize backend selector.

        Parameters
        ----------
        numba_threshold : int
            Minimum total elements to prefer numba over numpy
        cupy_threshold : int
            Minimum total elements to prefer cupy over numba
        polars_threshold : int
            Minimum total elements to prefer polars over numpy for dataframe ops
        auto_benchmark : bool
            If True, run benchmarks on first use to calibrate thresholds
        """
        self.numba_threshold = numba_threshold
        self.cupy_threshold = cupy_threshold
        self.polars_threshold = polars_threshold
        self.auto_benchmark = auto_benchmark

        self._capabilities = self._detect_capabilities()
        self._benchmark_cache: Dict[str, Dict[BackendType, float]] = {}

        if auto_benchmark and self._capabilities[BackendType.NUMBA].available:
            self._run_auto_benchmark()

    def _detect_capabilities(self) -> Dict[BackendType, BackendCapabilities]:
        """Detect available backends and their capabilities."""
        caps = {}

        # NumPy (always available)
        import numpy
        caps[BackendType.NUMPY] = BackendCapabilities(
            backend=BackendType.NUMPY,
            available=True,
            version=numpy.__version__,
        )

        # Numba
        try:
            import numba
            caps[BackendType.NUMBA] = BackendCapabilities(
                backend=BackendType.NUMBA,
                available=True,
                supports_jit=True,
                version=numba.__version__,
            )
        except ImportError:
            caps[BackendType.NUMBA] = BackendCapabilities(
                backend=BackendType.NUMBA,
                available=False,
            )

        # CuPy
        try:
            import cupy as cp
            try:
                device = cp.cuda.Device(0)
                device_info = f"GPU {device.id}: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}"
                caps[BackendType.CUPY] = BackendCapabilities(
                    backend=BackendType.CUPY,
                    available=True,
                    supports_gpu=True,
                    version=cp.__version__,
                    device_info=device_info,
                )
            except (cp.cuda.runtime.CUDARuntimeError, AttributeError):
                caps[BackendType.CUPY] = BackendCapabilities(
                    backend=BackendType.CUPY,
                    available=False,
                )
        except ImportError:
            caps[BackendType.CUPY] = BackendCapabilities(
                backend=BackendType.CUPY,
                available=False,
            )

        # Polars
        try:
            import polars
            caps[BackendType.POLARS] = BackendCapabilities(
                backend=BackendType.POLARS,
                available=True,
                supports_dataframe=True,
                version=polars.__version__,
            )
        except ImportError:
            caps[BackendType.POLARS] = BackendCapabilities(
                backend=BackendType.POLARS,
                available=False,
            )

        return caps

    def get_capabilities(self) -> Dict[BackendType, BackendCapabilities]:
        """Return detected backend capabilities."""
        return self._capabilities.copy()

    def select_for_cs_rank(
        self,
        T: int,
        N: int,
    ) -> BackendType:
        """
        Select optimal backend for cross-sectional ranking.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of assets

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N

        # Polars for large cross-sections
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
        ):
            return BackendType.POLARS

        # Numba for medium datasets
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        # NumPy default (bottleneck accelerated)
        return BackendType.NUMPY

    def select_for_cs_zscore(
        self,
        T: int,
        N: int,
    ) -> BackendType:
        """
        Select optimal backend for cross-sectional z-score.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of assets

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N

        # CuPy for very large datasets
        if (
            total_elements >= self.cupy_threshold
            and self._capabilities[BackendType.CUPY].available
        ):
            return BackendType.CUPY

        # Polars for large datasets
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
        ):
            return BackendType.POLARS

        # NumPy default (vectorized)
        return BackendType.NUMPY

    def select_for_rolling(
        self,
        T: int,
        N: int,
        window: int,
    ) -> BackendType:
        """
        Select optimal backend for rolling operations.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of series
        window : int
            Window size

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N

        # CuPy first for very large rolling workloads — checked before numba
        # so the numba branch below can't shadow it (previously the numba
        # check returned first whenever numba was installed, making the CuPy
        # branch reachable only when numba was absent).
        if (
            total_elements >= self.cupy_threshold
            and self._capabilities[BackendType.CUPY].available
        ):
            return BackendType.CUPY

        # Numba is typically best for rolling operations
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        return BackendType.NUMPY

    def select_for_neutralization(
        self,
        T: int,
        N: int,
        n_features: int,
    ) -> BackendType:
        """
        Select optimal backend for neutralization.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of assets
        n_features : int
            Number of features to neutralize against

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N * n_features

        # Polars for large datasets (good at groupby + regression)
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
        ):
            return BackendType.POLARS

        # Numba for medium datasets
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        return BackendType.NUMPY

    def _run_auto_benchmark(self):
        """Run automatic benchmarks to calibrate thresholds."""
        print("Running backend benchmarks...")

        # Small benchmark for rolling operations
        T, N = 500, 1000
        window = 20
        data = np.random.randn(T, N)

        self._benchmark_cache["rolling"] = {}

        # Numpy baseline
        start = time.perf_counter()
        from factor_preprocess.kernels.fast import fast_rolling_mean
        fast_rolling_mean(data, window)
        numpy_time = time.perf_counter() - start
        self._benchmark_cache["rolling"][BackendType.NUMPY] = numpy_time

        print(f"  NumPy rolling: {numpy_time*1000:.2f}ms")

        # Numba comparison
        if self._capabilities[BackendType.NUMBA].available:
            from factor_preprocess.kernels.fast import numba_rolling_mean
            start = time.perf_counter()
            numba_rolling_mean(data, window)
            numba_time = time.perf_counter() - start
            self._benchmark_cache["rolling"][BackendType.NUMBA] = numba_time
            speedup = numpy_time / numba_time
            print(f"  Numba rolling: {numba_time*1000:.2f}ms (speedup: {speedup:.1f}x)")


def get_backend_capabilities() -> Dict[BackendType, BackendCapabilities]:
    """
    Get capabilities of all available backends.

    Returns
    -------
    dict
        Mapping of backend type to capabilities

    Examples
    --------
    >>> caps = get_backend_capabilities()
    >>> if caps[BackendType.NUMBA].available:
    ...     print(f"Numba version: {caps[BackendType.NUMBA].version}")
    """
    selector = BackendSelector()
    return selector.get_capabilities()

def benchmark_backends(
    operation: str = "rolling",
    T: int = 500,
    N: int = 1000,
    window: int = 20,
    n_runs: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark available backends for a specific operation.

    Parameters
    ----------
    operation : str
        Operation to benchmark: "rolling", "cs_rank", "cs_zscore"
    T : int
        Number of time periods
    N : int
        Number of assets/series
    window : int
        Window size (for rolling operations)
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    dict
        Benchmark results with timing and speedup information

    Examples
    --------
    >>> results = benchmark_backends(operation="rolling", T=500, N=1000)
    >>> for backend, stats in results.items():
    ...     print(f"{backend}: {stats['mean_time']*1000:.2f}ms (speedup: {stats['speedup']:.1f}x)")
    """
    results = {}

    # Generate test data
    np.random.seed(42)
    data = np.random.randn(T, N)

    selector = BackendSelector()
    caps = selector.get_capabilities()

    if operation == "rolling":
        from factor_preprocess.kernels.fast import fast_rolling_mean, numba_rolling_mean

        # Benchmark NumPy
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fast_rolling_mean(data, window)
            times.append(time.perf_counter() - start)

        numpy_time = np.mean(times)
        results["numpy"] = {
            "mean_time": numpy_time,
            "std_time": np.std(times),
            "speedup": 1.0,
            "available": True,
        }

        # Benchmark Numba
        if caps[BackendType.NUMBA].available:
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                numba_rolling_mean(data, window)
                times.append(time.perf_counter() - start)

            numba_time = np.mean(times)
            results["numba"] = {
                "mean_time": numba_time,
                "std_time": np.std(times),
                "speedup": numpy_time / numba_time,
                "available": True,
            }
        else:
            results["numba"] = {
                "available": False,
                "note": "Numba not installed",
            }

    elif operation == "cs_rank":
        from factor_preprocess.kernels.fast import fast_cs_rank

        # Benchmark NumPy
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fast_cs_rank(data, axis=-1)
            times.append(time.perf_counter() - start)

        numpy_time = np.mean(times)
        results["numpy"] = {
            "mean_time": numpy_time,
            "std_time": np.std(times),
            "speedup": 1.0,
            "available": True,
        }

    elif operation == "cs_zscore":
        from factor_preprocess.kernels.fast import fast_cs_zscore

        # Benchmark NumPy
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fast_cs_zscore(data, axis=-1)
            times.append(time.perf_counter() - start)

        numpy_time = np.mean(times)
        results["numpy"] = {
            "mean_time": numpy_time,
            "std_time": np.std(times),
            "speedup": 1.0,
            "available": True,
        }

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return results
