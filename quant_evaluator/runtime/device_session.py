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
        self._staged_holding_returns: Dict[str, Any] = {}
        self._staged_trade_eligibility: Dict[str, Any] = {}
        self._intermediates: Dict[str, Any] = {}
        self._oom_retries = 0
        self._final_tile = 0
        self._closed = False
        self._peak_vram = 0.0
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._reuse_count = 0
        self._storage_dtypes = {}

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> "DeviceEvaluationSession":
        self._open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self) -> None:
        if self.policy.required_capabilities:
            raise UnsupportedBackendCapability(f"mandatory capabilities unsupported: {self.policy.required_capabilities}")
        if len(self.policy.device_ids) != 1:
            raise UnsupportedBackendCapability("one session supports exactly one device")
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
        # Effective transfer is synchronous. Merely allocating streams does
        # not implement pinned/double-buffer lifetime ownership.
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
                self._staged_holding_returns.clear()
                self._staged_trade_eligibility.clear()
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
        dtype = cp.float64 if self.policy.precision_policy in (PrecisionPolicy.GPU_FP64, PrecisionPolicy.REFERENCE_FP64) else None
        dev = cp.asarray(values, dtype=dtype)
        cp.cuda.get_current_stream().synchronize()
        self._storage_dtypes["factors"] = str(dev.dtype)
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
        dtype = self._cp.float64 if self.policy.precision_policy in (PrecisionPolicy.GPU_FP64, PrecisionPolicy.REFERENCE_FP64) else None
        dev = self._cp.asarray(values, dtype=dtype)
        self._cp.cuda.get_current_stream().synchronize()
        self._storage_dtypes["labels"] = str(dev.dtype)
        self._staged_labels[target_id] = dev
        self._h2d_bytes += dev.nbytes
        return dev

    def stage_holding_returns(self, panel):
        from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel
        if not isinstance(panel, HoldingReturnPanel):
            raise TypeError("stage_holding_returns requires HoldingReturnPanel, never forward labels")
        dev = self._cp.asarray(panel.values, dtype=self._cp.float64)
        self._staged_holding_returns[panel.content_hash] = dev
        self._h2d_bytes += dev.nbytes
        return panel.content_hash

    def stage_trade_eligibility(self, panel):
        from quant_evaluator.contracts.portfolio_inputs import TradeEligibilityPanel
        if not isinstance(panel, TradeEligibilityPanel):
            raise TypeError("stage_trade_eligibility requires TradeEligibilityPanel")
        dev = self._cp.stack([self._cp.asarray(getattr(panel, name))
                             for name in ("can_buy", "can_sell", "borrowable", "coverable")])
        self._staged_trade_eligibility[panel.content_hash] = dev
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
        value = self._intermediates.get(key)
        if value is not None:
            self._reuse_count += 1
        return value

    def has_intermediate(self, key: str) -> bool:
        return key in self._intermediates

    # -- tiling / OOM ---------------------------------------------------
    def workspace_budget(self, *, output_bytes: int = 0) -> int:
        """Admit bounded kernel scratch after live allocations and outputs.

        Cached free pool blocks are reusable, not extra live input. The pool
        limit remains the hard allocation backstop; OOM is handled by retile.
        """
        if isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes < 0:
            raise ValueError("output_bytes must be a nonnegative integer")
        remaining = int(self._vram_budget) - self._pool.used_bytes() - output_bytes
        if remaining <= 0:
            raise MemoryError("device output admission exceeds session budget")
        return min(1 << 30, max(1, int(remaining * .8)))

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
        #   argsort scratch ~ 4 x (T,ftile,N) actual int64 indices
        #   + bounded per-chunk rank scratch (~1GB, see kernels/gpu/rank.py)
        # Conservative admission estimate, not a measured peak for this call.
        if self.policy.precision_policy in (PrecisionPolicy.GPU_FP64, PrecisionPolicy.REFERENCE_FP64):
            dtype_bytes = max(dtype_bytes, 8)
        factor_bytes = T * ftile * N * dtype_bytes
        label_bytes = T * N * 8  # float64 label
        rank_tensors = 12 * T * ftile * N * 8  # float64 rank intermediates
        argsort_scratch = 4 * T * ftile * N * 8  # int64 index scratch allowance
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
        pool = getattr(self, '_pool', None)
        reserved = pool.total_bytes() if pool is not None else 0
        active = pool.used_bytes() if pool is not None else 0
        self._peak_vram = max(self._peak_vram, reserved)
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
            "intermediate_reuse_count": self._reuse_count,
            "intermediate_entries": len(self._intermediates),
            "pool_reserved_peak_bytes": self._peak_vram,
            "pool_active_bytes_at_receipt": active,
            "pool_reserved_bytes_at_receipt": reserved,
            "storage_dtypes": dict(self._storage_dtypes),
            "compute_dtype": "per_metric_implementation",
            "accumulator_dtype": "per_metric_implementation",
            "requested_transfer": {"async": self.policy.async_transfer, "pinned": self.policy.pinned_host_memory, "double_buffer": self.policy.double_buffer},
            "effective_transfer": "synchronous",
            "effective_pinned": False,
            "effective_buffers": 1,
            "transfer_fallback_reason": "async pinned double-buffer pipeline is not implemented",
        }
