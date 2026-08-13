"""
Backend selector for automatic backend selection based on data size and capabilities.

Automatically selects the optimal backend (numpy, numba, cupy, polars) based on:
- Available hardware/libraries
- Data shape and size
- Operation type
- Benchmark results
"""

from typing import Optional, Dict, Any, Tuple, Callable
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
    >>> backend = selector.select_for_ic(T=252, N=5000, F=10000)
    >>> print(backend)  # "numba" or "cupy" if available

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

    def select_for_ic(
        self,
        T: int,
        N: int,
        F: int,
        method: str = "pearson",
    ) -> BackendType:
        """
        Select optimal backend for IC computation.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of assets
        F : int
            Number of factors
        method : str
            Correlation method ("pearson" or "spearman")

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N * F

        # CuPy for very large batches (if available)
        if (
            total_elements >= self.cupy_threshold
            and self._capabilities[BackendType.CUPY].available
        ):
            return BackendType.CUPY

        # Polars for large batches with many groups (efficient groupby)
        # Polars is especially good for IC because it's groupby-heavy
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
            and F >= 100  # Many factors = many groups
        ):
            return BackendType.POLARS

        # Numba for medium-to-large batches
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        # NumPy for small batches (reference implementation)
        return BackendType.NUMPY

    def select_for_quantile(
        self,
        T: int,
        N: int,
        F: int,
        n_quantiles: int = 5,
    ) -> BackendType:
        """
        Select optimal backend for quantile binning.

        Parameters
        ----------
        T : int
            Number of time periods
        N : int
            Number of assets
        F : int
            Number of factors
        n_quantiles : int
            Number of quantiles

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = T * N * F

        # CuPy for very large datasets
        if (
            total_elements >= self.cupy_threshold
            and self._capabilities[BackendType.CUPY].available
        ):
            return BackendType.CUPY

        # Polars for large datasets with many factors (efficient ranking)
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
            and F >= 50
        ):
            return BackendType.POLARS

        # Numba for medium datasets
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        # NumPy default
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

        # Numba is typically best for rolling operations
        if (
            total_elements >= self.numba_threshold
            and self._capabilities[BackendType.NUMBA].available
        ):
            return BackendType.NUMBA

        # CuPy for very large rolling windows
        if (
            total_elements >= self.cupy_threshold
            and self._capabilities[BackendType.CUPY].available
        ):
            return BackendType.CUPY

        return BackendType.NUMPY

    def select_for_dataframe(
        self,
        n_rows: int,
        n_cols: int,
    ) -> BackendType:
        """
        Select optimal backend for DataFrame operations.

        Parameters
        ----------
        n_rows : int
            Number of rows
        n_cols : int
            Number of columns

        Returns
        -------
        BackendType
            Selected backend
        """
        total_elements = n_rows * n_cols

        # Polars for large dataframes
        if (
            total_elements >= self.polars_threshold
            and self._capabilities[BackendType.POLARS].available
        ):
            return BackendType.POLARS

        return BackendType.NUMPY

    def _run_auto_benchmark(self):
        """Run automatic benchmarks to calibrate thresholds."""
        # Small benchmark to establish performance characteristics
        print("Running backend benchmarks...")

        # Benchmark IC computation
        T, N, F = 100, 1000, 100
        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        self._benchmark_cache["ic"] = {}

        # Numpy baseline
        start = time.perf_counter()
        from quant_evaluator.kernels.fast import fast_ic_batch
        fast_ic_batch(factors, labels)
        numpy_time = time.perf_counter() - start
        self._benchmark_cache["ic"][BackendType.NUMPY] = numpy_time

        print(f"  NumPy IC: {numpy_time*1000:.2f}ms")

        # TODO: Add numba and cupy benchmarks when implemented


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
    >>> if caps[BackendType.CUPY].available:
    ...     print(f"GPU: {caps[BackendType.CUPY].device_info}")
    """
    selector = BackendSelector()
    return selector.get_capabilities()


def benchmark_backends(
    operation: str = "ic",
    T: int = 252,
    N: int = 3000,
    F: int = 1000,
    n_runs: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark available backends for a specific operation.

    Parameters
    ----------
    operation : str
        Operation to benchmark: "ic", "quantile", "rolling"
    T : int
        Number of time periods
    N : int
        Number of assets/series
    F : int
        Number of factors (for ic/quantile)
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    dict
        Benchmark results with timing and speedup information

    Examples
    --------
    >>> results = benchmark_backends(operation="ic", T=252, N=3000, F=1000)
    >>> for backend, stats in results.items():
    ...     print(f"{backend}: {stats['mean_time']*1000:.2f}ms (speedup: {stats['speedup']:.1f}x)")
    """
    results = {}

    # Generate test data
    np.random.seed(42)
    factors = np.random.randn(T, N, F)
    labels = np.random.randn(T, N)

    selector = BackendSelector()
    caps = selector.get_capabilities()

    if operation == "ic":
        from quant_evaluator.kernels.fast import fast_ic_batch

        # Benchmark NumPy
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fast_ic_batch(factors, labels)
            times.append(time.perf_counter() - start)

        numpy_time = np.mean(times)
        results["numpy"] = {
            "mean_time": numpy_time,
            "std_time": np.std(times),
            "speedup": 1.0,
            "available": True,
        }

        # Benchmark CuPy if available
        if caps[BackendType.CUPY].available:
            try:
                from quant_evaluator.backends.cupy_backend import create_gpu_backend
                gpu_backend = create_gpu_backend()
                if gpu_backend:
                    times = []
                    for _ in range(n_runs):
                        start = time.perf_counter()
                        gpu_backend.fast_ic_batch_gpu(factors, labels)
                        times.append(time.perf_counter() - start)

                    cupy_time = np.mean(times)
                    results["cupy"] = {
                        "mean_time": cupy_time,
                        "std_time": np.std(times),
                        "speedup": numpy_time / cupy_time,
                        "available": True,
                    }
                    gpu_backend.clear_memory_pool()
            except Exception as e:
                results["cupy"] = {
                    "available": False,
                    "error": str(e),
                }

    elif operation == "rolling":
        # Benchmark rolling operations
        window = 20
        data = np.random.randn(T, N)

        from quant_evaluator.kernels.fast import fast_ic_batch
        # Use existing kernels as proxy
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            # Simple rolling mean as benchmark
            result = np.full_like(data, np.nan)
            for t in range(window - 1, T):
                result[t] = np.nanmean(data[t - window + 1:t + 1], axis=0)
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
