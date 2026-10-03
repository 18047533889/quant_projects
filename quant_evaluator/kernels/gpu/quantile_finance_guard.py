"""Unintegrated bounded-Q finance-scale numeric risk guard prototype.

The kernel preserves the existing risk_guard ABI and computes O(N+Q) scans per
row for Q<=32. Per-thread bucket aggregates are statically bounded; their actual
RawKernel local-memory size must be queried after compile and admitted by the
caller using local_workspace_bytes(). No host data copies or CPU fallback exist.
"""
from __future__ import annotations

from numbers import Integral

MAX_FINANCE_QUANTILES = 32

FINANCE_RISK_GUARD_SOURCE = r"""
extern "C" __global__ void finance_risk_guard(
    const int* __restrict__ bucket, const double* __restrict__ data,
    const long long* __restrict__ counts, const double* __restrict__ means,
    unsigned char* __restrict__ risk, int* __restrict__ error_flag,
    long long nrows, long long ncols, long long nq, long long min_assets) {
  long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=nrows) return;
  if(nq<1 || nq>32 || ncols<1 || ncols>2147483647LL || min_assets<1) {
    atomicExch(error_flag,1);
    return;
  }

  const double eps=2.2204460492503130808472633361816e-16;
  const double tiny=2.2250738585072013830902327173324e-308;
  const double max_double=1.797693134862315708145274237317e308;
  double bucket_max[32];
  double bucket_scaled_sum[32];
  long long bucket_finite_count[32];
  int bucket_has_subnormal[32];
  for(int q=0;q<32;q++) {
    bucket_max[q]=0.0;
    bucket_scaled_sum[q]=0.0;
    bucket_finite_count[q]=0;
    bucket_has_subnormal[q]=0;
  }

  // Pass one: establish the row scale and validate all bucket IDs.
  double row_max=0.0;
  int row_has_subnormal=0;
  for(long long j=0;j<ncols;j++) {
    int b=bucket[row*ncols+j];
    if(b < -1 || b >= nq) atomicExch(error_flag,1);
    double v=data[row*ncols+j];
    if(!isfinite(v)) continue;
    double a=fabs(v);
    if(a>row_max) row_max=a;
    if(a>0.0 && a<tiny) row_has_subnormal=1;
  }

  // Pass two: full-row scaled absolute sum plus online per-bucket
  // max/scaled-sum/count aggregates. Online rescaling avoids raw-sum overflow.
  double row_scaled_sum=0.0;
  int mixed_scale=0;
  for(long long j=0;j<ncols;j++) {
    int b=bucket[row*ncols+j];
    double v=data[row*ncols+j];
    if(!isfinite(v)) continue;
    double a=fabs(v);
    if(row_max>0.0) {
      double term=a/row_max;
      if(a>0.0 && term<tiny) mixed_scale=1;
      row_scaled_sum+=term;
    }
    if(b<0 || b>=nq) continue;
    bucket_finite_count[b]++;
    if(a>0.0 && a<tiny) bucket_has_subnormal[b]=1;
    if(a==0.0) continue;
    double old_max=bucket_max[b];
    if(old_max==0.0) {
      bucket_max[b]=a;
      bucket_scaled_sum[b]=1.0;
    } else if(a>old_max) {
      double ratio=old_max/a;
      if(ratio<tiny) mixed_scale=1;
      bucket_scaled_sum[b]=bucket_scaled_sum[b]*ratio+1.0;
      bucket_max[b]=a;
    } else {
      double ratio=a/old_max;
      if(ratio<tiny) mixed_scale=1;
      bucket_scaled_sum[b]+=ratio;
    }
  }

  double neps=(double)ncols*eps;
  double round_factor=1.0+16.0*eps;
  double gamma_n=(neps/(1.0-neps))*round_factor;
  // The global sum performs one division and one addition per observation.
  // Inflate the computed positive sum to bound downward rounding.
  double global_ops=2.0*(double)ncols*eps;
  double gamma_global=(global_ops/(1.0-global_ops))*round_factor;
  double global_sum_upper=(row_scaled_sum/(1.0-gamma_global))*round_factor;
  int overflow=(row_max>0.0 &&
      row_max>=max_double/(fmax(global_sum_upper,1.0)*round_factor));
  int global_subnormal=(row_max>0.0 && row_max<tiny);
  int row_exact=overflow || row_has_subnormal || global_subnormal || mixed_scale;

  for(long long q=0;q<nq;q++) {
    long long id=row*nq+q;
    long long declared=counts[id];
    if(declared<0 || declared>ncols) {
      atomicExch(error_flag,1);
      risk[id]=0;
      continue;
    }
    if(bucket_finite_count[q]!=declared) atomicExch(error_flag,1);
    if(declared<min_assets || declared==0) {
      risk[id]=0;
      continue;
    }

    double mean=means[id];
    int mean_subnormal=(isfinite(mean) && mean!=0.0 && fabs(mean)<tiny);
    int bad=(!isfinite(mean) || row_exact || mean_subnormal);
    if(!bad && row_max>0.0) {
      // Preserve the full-N gamma bound. Inflation protects against downward
      // rounding in the global scaled sum and error-bound arithmetic.
      double coarse_error=((2.0*gamma_n*row_max)*global_sum_upper/
                           (double)declared)*round_factor;
      if(!isfinite(coarse_error) || coarse_error>1.0e-12) bad=1;
      // As in the established guard, only ordinary finance-scale rows refine;
      // overflow, mixed-scale, subnormal and large-scale cases remain exact.
      if(bad && !overflow && !mixed_scale && !row_has_subnormal &&
         !global_subnormal && !mean_subnormal && row_max<=1.0 &&
         isfinite(mean)) {
        if(bucket_has_subnormal[q]) {
          // Keep subnormal buckets on the exact path.
        } else if(bucket_max[q]==0.0) {
          bad=0;
        } else {
          double c_eps=3.0*(double)declared*eps;
          double gamma_bucket=(c_eps/(1.0-c_eps))*round_factor;
          double scaled_upper=(bucket_scaled_sum[q]/(1.0-gamma_bucket))*
                              round_factor;
          double bucket_error=((2.0*gamma_n*bucket_max[q])*scaled_upper/
                               (double)declared)*round_factor;
          bad=(!isfinite(bucket_error) || bucket_error>1.0e-12);
        }
      }
    }
    risk[id]=(unsigned char)bad;
  }
}
"""


def validate_quantile_count(n_quantiles: int) -> int:
    if isinstance(n_quantiles, bool) or not isinstance(n_quantiles, int):
        raise ValueError("n_quantiles must be a builtin integer")
    if not 1 <= n_quantiles <= MAX_FINANCE_QUANTILES:
        raise ValueError("finance risk guard supports quantile counts in [1, 32]")
    return n_quantiles


def compile_finance_risk_guard(cp):
    """Compile the prototype RawKernel; CuPy is an explicit required argument."""
    if not callable(getattr(cp, "RawKernel", None)):
        raise TypeError("a CuPy-compatible RawKernel factory is required")
    kernel = cp.RawKernel(
        FINANCE_RISK_GUARD_SOURCE,
        "finance_risk_guard",
        options=("--std=c++11",),
    )
    kernel.compile()
    return kernel


def local_workspace_bytes(kernel, *, rows: int, block_size: int = 128) -> dict[str, int]:
    """Report actual compiled local-memory use for a rounded one-thread-per-row launch.

    This excludes user-provided risk/count/output arrays and kernel parameters;
    callers should admit this reported local allocation in addition to their
    global scratch estimates before launching the kernel.
    """
    for name, value in (("rows", rows), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative builtin integer")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    kernel.compile()
    attrs = kernel.attributes
    per_thread = attrs.get("local_size_bytes", attrs.get("localSizeBytes"))
    if per_thread is None:
        raise RuntimeError("CuPy did not report RawKernel local_size_bytes")
    if isinstance(per_thread, bool) or not isinstance(per_thread, Integral) or per_thread < 0:
        raise RuntimeError("RawKernel local_size_bytes is invalid")
    per_thread = int(per_thread)
    launched = ((rows + block_size - 1) // block_size) * block_size
    return {
        "local_size_bytes_per_thread": per_thread,
        "launched_threads": launched,
        "local_workspace_bytes": per_thread * launched,
    }
