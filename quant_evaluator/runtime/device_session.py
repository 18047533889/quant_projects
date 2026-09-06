"""Device evaluation session (spec §5, §7, §8, §41).

Owns the GPU device, pinned host memory, streams, VRAM budget, factor
tiling, and OOM retile.  Factors/labels are staged once and reused across
metrics; device intermediates live in a session-scoped ephemeral cache.

This module imports CuPy lazily so a CPU-only environment imports cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from quant_evaluator.contracts.backend_policy import (
    GPUExecutionPolicy,
    PrecisionPolicy,
)

logger = logging.getLogger(__name__)


class UnsupportedBackendCapability(RuntimeError):
    """Raised when a mandatory metric has no CUDA implementation (CUDA_STRICT)."""


class DeviceEvaluationSession:
    """A session that keeps factors/labels resident on the GPU.

    Usage (spec §5):

        with DeviceEvaluationSession(policy) as session:
            factor_dev = session.stage_factors(factor_values, factor_ids)
            label_dev = session.stage_labels(label_values)
            plan = planner.compile(...)
            outputs = executor.run(plan, factor_dev, label_dev)
            return session.materialize_outputs(outputs)
    """

    def __init__(self, policy: Optional[GPUExecutionPolicy] = None) -> None:
        self.policy = policy or GPUExecutionPolicy()
        self._cp = None
        self._device = None
        self._streams: List[Any] = []
        self._pinned: List[Any] = []
        self._staged_factors: Dict[str, Any] = {}
        self._staged_labels: Dict[str, Any] = {}
        self._intermediates: Dict[str, Any] = {}
        self._oom_retries = 0
        self._final_tile = 0
        self._closed = False
        self._peak_vram = 0.0
        self._h2d_bytes = 0
        self._d2h_bytes = 0

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> "DeviceEvaluationSession":
        self._open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self) -> None:
        import cupy as cp  # lazy

        self._cp = cp
        dev_id = self.policy.device_ids[0]
        self._device = cp.cuda.Device(dev_id)
        self._device.use()
        free, total = cp.cuda.runtime.memGetInfo()
        self._free_vram = float(free)
        self._total_vram = float(total)
        self._vram_budget = self._free_vram * self.policy.max_vram_fraction
        # session-scoped memory pool with a soft cap
        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=int(self._vram_budget))
        self._pool = pool
        if self.policy.async_transfer:
            self._streams = [cp.cuda.Stream() for _ in range(3)]
        # Process-global set_allocator races when independent evaluation
        # workers overlap. CuPy's context is thread-local and nests safely.
        self._allocator_scope = cp.cuda.using_allocator(pool.malloc)
        self._allocator_scope.__enter__()
        logger.info(
            "DeviceEvaluationSession open: device=%s free_vram=%.1fGB budget=%.1fGB",
            dev_id, self._free_vram / 1e9, self._vram_budget / 1e9,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._cp is not None:
            try:
                self._cp.cuda.Device().synchronize()
                # Drop every session-owned device reference before asking the
                # pool to release blocks. free_all_blocks cannot free arrays
                # that are still retained in these caches.
                self._staged_factors.clear()
                self._staged_labels.clear()
                self._intermediates.clear()
                self._streams.clear()
                self._pinned.clear()
                self._pool.free_all_blocks()
            except Exception:
                pass
            finally:
                scope = getattr(self, "_allocator_scope", None)
                if scope is not None:
                    scope.__exit__(None, None, None)
                    self._allocator_scope = None
        self._closed = True

    # -- staging -------------------------------------------------------
    def stage_factors(self, values, factor_ids, layout: str = "T,F,N"):
        """Upload factor values once; returns a device array."""
        cp = self._cp
        dev = cp.asarray(values)
        if layout == "T,N,F":
            # transpose to GPU-friendly (T, F, N)
            dev = cp.transpose(dev, (0, 2, 1))
        self._staged_factors["__all__"] = dev
        self._h2d_bytes += dev.nbytes
        return dev

    def release_factor_tile(self) -> None:
        """Release only tile-owned allocations; labels stay resident."""
        self._peak_vram = max(self._peak_vram, self._pool.total_bytes())
        self._staged_factors.pop("__all__", None)
        self._intermediates.clear()
        self._pool.free_all_blocks()

    def stage_labels(self, values, target_id: str = "next_ret"):
        dev = self._cp.asarray(values)
        self._staged_labels[target_id] = dev
        self._h2d_bytes += dev.nbytes
        return dev

    def stage_context(self, name: str, values):
        dev = self._cp.asarray(values)
        self._staged_factors[f"ctx:{name}"] = dev
        self._h2d_bytes += dev.nbytes
        return dev

    # -- intermediates --------------------------------------------------
    def put_intermediate(self, key: str, value: Any) -> None:
        self._intermediates[key] = value

    def get_intermediate(self, key: str) -> Any:
        return self._intermediates.get(key)

    def has_intermediate(self, key: str) -> bool:
        return key in self._intermediates

    # -- tiling / OOM ---------------------------------------------------
    def estimate_tile(self, metric_plan, T: int, N: int, dtype_bytes: int) -> int:
        """Pick an initial factor tile from a candidate list (spec §7)."""
        if not hasattr(self, "_vram_budget") or self._vram_budget is None:
            self._open()  # ensure budget is computed (session may not be open yet)
        for tile in (128, 64, 32, 16, 8, 4, 2, 1):
            est = self._estimate_working_set(metric_plan, T, N, tile, dtype_bytes)
            if est <= self._vram_budget:
                self._final_tile = tile
                return tile
        raise MemoryError(
            "GPU memory budget cannot fit a single factor working set; "
            "reduce the time/asset panel or increase max_vram_fraction"
        )

    def _estimate_working_set(self, metric_plan, T, N, ftile, dtype_bytes) -> int:
        # Realistic working set for the Spearman rank family (spec §57),
        # calibrated against the real 500-factor benchmark on L20:
        #   factor tile (T,ftile,N) in input dtype
        #   label (T,N) float64
        #   x_rank / y_rank / rx / ry: 4 x (T,ftile,N) float64 rank tensors
        #   argsort scratch (thrust) ~ 4 x (T,ftile,N) int32
        #   + bounded per-chunk rank scratch (~1GB, see kernels/gpu/rank.py)
        # The 12x float64 multiplier on rank tensors + argsort scratch is
        # calibrated so F=32 (~32GB peak) is rejected and F=16 (~21GB) fits.
        factor_bytes = T * ftile * N * dtype_bytes
        label_bytes = T * N * 8  # float64 label
        rank_tensors = 12 * T * ftile * N * 8  # float64 rank intermediates
        argsort_scratch = 4 * T * ftile * N * 4  # thrust int32 scratch
        chunk_scratch = 1 << 30  # 1 GiB bounded rank chunk
        return factor_bytes + label_bytes + rank_tensors + argsort_scratch + chunk_scratch

    def retile_on_oom(self, current_tile: int) -> int:
        """Halve the tile on OOM; record retry (spec §7)."""
        self._oom_retries += 1
        nxt = max(current_tile // 2, 1)
        self._final_tile = nxt
        logger.warning("OOM retile: %d -> %d (retries=%d)", current_tile, nxt, self._oom_retries)
        return nxt

    # -- observability --------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        return {
            "backend_requested": "cuda",
            "backend_used": "cuda",
            "gpu_device": self.policy.device_ids[0],
            "precision": self.policy.precision_policy.value,
            "factor_tile_size": self._final_tile,
            "oom_retries": self._oom_retries,
            "h2d_bytes": self._h2d_bytes,
            "d2h_bytes": self._d2h_bytes,
            "peak_vram": self._peak_vram,
            "vram_budget_bytes": getattr(self, "_vram_budget", None),
            "intermediate_reuse_count": len(self._intermediates),
        }
