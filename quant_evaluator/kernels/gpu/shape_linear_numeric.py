"""Device-only exact repair for linear quantile-shape summaries."""
from __future__ import annotations

from numbers import Integral
from typing import Any

from .dyadic_accumulator_cuda import DYADIC_DEVICE_SOURCE

_DEFAULT_WORKSPACE_BYTES = 1 << 30
_THREADS = 32
_GUARD_THREADS = 128
_INT32_MAX = 2147483647
_METRIC_IDS = {
    "curvature": 0,
    "spread": 1,
    "tail_asymmetry": 2,
    "extreme_cliff": 3,
    "top_cliff": 4,
    "bottom_cliff": 5,
}

_RISK_GUARD = r"""
extern "C" __global__ void shape_linear_risk(
    const double* __restrict__ values, unsigned char* __restrict__ risk,
    long long nq, long long nfeatures, int metric, int* __restrict__ error_flag) {
  long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=nfeatures) return;
  double l1=4.0;
  if(metric==0) l1=4.0*(nq>2 ? (double)(nq-2) : 1.0);
  else if(metric==1) l1=2.0*(nq>1 ? (double)(nq-1) : 1.0);
  else if(metric==4 || metric==5) l1=2.0;
  double max_abs=0.0;
  int subnormal=0;
  const double tiny=2.2250738585072013830902327173324e-308;
  for(long long q=0;q<nq;q++) {
    double v=values[q*nfeatures+f];
    if(!isfinite(v)) continue;
    double a=fabs(v);
    if(a>max_abs) max_abs=a;
    if(a>0.0 && a<tiny*(l1+2.0)) subnormal=1;
  }
  const double eps=2.2204460492503130808472633361816e-16;
  const double error_bound=1.0e-12/(2.0*eps*(l1+2.0));
  const double max_finite=1.797693134862315708145274237317e308;
  int overflow_bound=(max_abs>0.0 && max_abs>max_finite/l1);
  risk[f]=(unsigned char)(subnormal || max_abs>=error_bound || overflow_bound);
}
"""

_SHAPE_EXACT = DYADIC_DEVICE_SOURCE + r"""
extern "C" __global__ void shape_linear_exact(
    const double* __restrict__ values, double* __restrict__ output,
    const long long* __restrict__ ids, int* __restrict__ error_flag,
    long long nq, long long nfeatures, long long nids, int metric) {
  long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=nids) return;
  long long f=ids[k];
  DyadicAccumulator accumulator;
  dyadic_init(&accumulator);
  long long denominator=0;
  long long weighted_terms=0;

  if(metric==0) {  // mean second difference, Q x F stencil
    for(long long q=1;q<nq-1;q++) {
      double lower=values[(q-1)*nfeatures+f];
      double middle=values[q*nfeatures+f];
      double upper=values[(q+1)*nfeatures+f];
      if(!isfinite(lower) || !isfinite(middle) || !isfinite(upper)) continue;
      dyadic_add(&accumulator,upper,1);
      dyadic_add(&accumulator,middle,-2);
      dyadic_add(&accumulator,lower,1);
      ++denominator;
      weighted_terms+=4;
    }
  } else if(metric==1) {  // mean absolute adjacent difference
    for(long long q=0;q<nq-1;q++) {
      double left=values[q*nfeatures+f];
      double right=values[(q+1)*nfeatures+f];
      if(!isfinite(left) || !isfinite(right)) continue;
      if(right>=left) {
        dyadic_add(&accumulator,right,1);
        dyadic_add(&accumulator,left,-1);
      } else {
        dyadic_add(&accumulator,left,1);
        dyadic_add(&accumulator,right,-1);
      }
      ++denominator;
      weighted_terms+=2;
    }
  } else if(metric==2) {  // last - 2*middle + first
    long long middle_index=nq/2;
    double first=values[f];
    double middle=values[middle_index*nfeatures+f];
    double last=values[(nq-1)*nfeatures+f];
    if(isfinite(first) && isfinite(middle) && isfinite(last)) {
      dyadic_add(&accumulator,last,1);
      dyadic_add(&accumulator,middle,-2);
      dyadic_add(&accumulator,first,1);
      denominator=1;
      weighted_terms=4;
    }
  } else if(metric==3) {  // average of top and bottom adjacent cliffs
    if(nq>=2) {
      double first=values[f];
      double second=values[nfeatures+f];
      double penultimate=values[(nq-2)*nfeatures+f];
      double last=values[(nq-1)*nfeatures+f];
      if(isfinite(first) && isfinite(second)
          && isfinite(penultimate) && isfinite(last)) {
        dyadic_add(&accumulator,last,1);
        dyadic_add(&accumulator,penultimate,-1);
        dyadic_add(&accumulator,second,1);
        dyadic_add(&accumulator,first,-1);
        denominator=2;
        weighted_terms=4;
      }
    }
  } else if(metric==4) {  // top cliff
    if(nq>=2) {
      double penultimate=values[(nq-2)*nfeatures+f];
      double last=values[(nq-1)*nfeatures+f];
      if(isfinite(penultimate) && isfinite(last)) {
        dyadic_add(&accumulator,last,1);
        dyadic_add(&accumulator,penultimate,-1);
        denominator=1;
        weighted_terms=2;
      }
    }
  } else if(metric==5) {  // bottom cliff
    if(nq>=2) {
      double first=values[f];
      double second=values[nfeatures+f];
      if(isfinite(first) && isfinite(second)) {
        dyadic_add(&accumulator,second,1);
        dyadic_add(&accumulator,first,-1);
        denominator=1;
        weighted_terms=2;
      }
    }
  } else {
    atomicExch(error_flag,1);
    return;
  }
  if(weighted_terms>2147483647LL) {
    atomicExch(error_flag,1);
    return;
  }
  output[f]=(denominator>0)
      ? dyadic_quotient_rne(&accumulator,(unsigned long long)denominator)
      : __longlong_as_double((long long)0x7ff8000000000000ULL);
}
"""


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - host without CUDA
        raise RuntimeError("CuPy is required for GPU shape-linear repair") from exc
    return cp


def _local_size_bytes(kernel: Any) -> int:
    kernel.compile()
    attrs = kernel.attributes
    value = attrs.get("local_size_bytes", attrs.get("localSizeBytes"))
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise RuntimeError("CuPy reported invalid RawKernel local_size_bytes")
    return int(value)


def repair_shape_linear_gpu(
    values: Any,
    output: Any,
    metric: str,
    *,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> Any:
    """Repair risky feature columns with exact dyadic linear arithmetic."""
    cp = _cupy()
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, Integral) or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")
    if not isinstance(metric, str) or metric not in _METRIC_IDS:
        raise ValueError(f"unsupported shape-linear metric: {metric!r}")
    if not isinstance(values, cp.ndarray) or not isinstance(output, cp.ndarray):
        raise TypeError("shape repair inputs and output must be CuPy device arrays")
    if values.dtype != cp.float64 or output.dtype != cp.float64:
        raise TypeError("shape repair values and output must be float64")
    if values.ndim != 2 or output.ndim != 1 or output.shape[0] != values.shape[1]:
        raise ValueError("shape repair requires values (Q,F) and output (F,)")
    if not values.flags.c_contiguous or not output.flags.c_contiguous:
        raise ValueError("shape repair arrays must be C-contiguous")
    current_device = cp.cuda.Device().id
    if values.device.id != current_device or output.device.id != current_device:
        raise ValueError("shape repair arrays must be on the current CUDA device")
    val_start = int(values.data.ptr)
    val_stop = val_start + int(values.nbytes)
    out_start = int(output.data.ptr)
    out_stop = out_start + int(output.nbytes)
    if val_start < out_stop and out_start < val_stop:
        raise ValueError("shape repair output must not overlap values")

    nq, nfeatures = map(int, values.shape)
    if nq > _INT32_MAX:
        raise ValueError("Q exceeds the int32 admission bound")
    if metric == "curvature" and 4 * max(0, nq - 2) > _INT32_MAX:
        raise ValueError("curvature weighted-term count exceeds int32max")
    if metric == "spread" and 2 * max(0, nq - 1) > _INT32_MAX:
        raise ValueError("spread weighted-term count exceeds int32max")
    if nfeatures == 0:
        return output

    guard = cp.RawKernel(_RISK_GUARD, "shape_linear_risk", options=("--std=c++11",))
    guard_local = _local_size_bytes(guard)
    guard_threads = ((nfeatures + _GUARD_THREADS - 1) // _GUARD_THREADS) * _GUARD_THREADS
    # Guard output, scalar error metadata and compiled guard local memory.
    guard_required = nfeatures * 8 + 4 + guard_local * guard_threads
    if guard_required > int(workspace_bytes):
        raise MemoryError(f"shape-linear risk guard requires {guard_required} workspace bytes")

    risk = cp.empty((nfeatures,), dtype=cp.uint8)
    error_flag = cp.zeros((), dtype=cp.int32)
    metric_id = _METRIC_IDS[metric]
    guard(((nfeatures + _GUARD_THREADS - 1) // _GUARD_THREADS,), (_GUARD_THREADS,),
          (values, risk, nq, nfeatures, metric_id, error_flag))
    risk_count = int(cp.sum(risk, dtype=cp.int64).item())
    if risk_count == 0:
        return output

    exact = cp.RawKernel(_SHAPE_EXACT, "shape_linear_exact", options=("--std=c++11",))
    exact_local = _local_size_bytes(exact)
    # Include worst-case IDs and nonzero scratch even when only one feature is risky.
    base_required = nfeatures * 48 + 4
    required = base_required + guard_local * guard_threads + exact_local * _THREADS
    if required > int(workspace_bytes):
        raise MemoryError(f"shape-linear exact repair requires {required} workspace bytes")
    ids = cp.nonzero(risk)[0].astype(cp.int64, copy=False)
    for start in range(0, risk_count, _THREADS):
        stop = min(start + _THREADS, risk_count)
        exact((1,), (_THREADS,),
              (values, output, ids[start:stop], error_flag, nq, nfeatures,
               stop - start, metric_id))
    if int(error_flag.item()):
        raise ValueError("shape-linear exact repair exceeded its admitted domain")
    return output


__all__ = ["repair_shape_linear_gpu"]
