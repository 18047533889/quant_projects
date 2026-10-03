"""Experimental GPU repair for numerically unsafe quantile bucket means.

The normal fast path remains the caller's existing CuPy reduction.  This module
only repairs buckets selected by a conservative on-device error/overflow guard.
It imports CuPy lazily and never copies data arrays to the host.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

_LIMBS = 68
_THREADS = 32
_FIXED_MEAN = r"""
extern "C" __global__ void fixed_mean(
    const int* __restrict__ bucket, const double* __restrict__ data,
    double* __restrict__ means, const long long* __restrict__ counts,
    const long long* __restrict__ ids, int* __restrict__ error_flag,
    long long nrows, long long ncols, long long nq, long long min_assets,
    long long nids) {
  long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=nids) return;
  long long id=ids[k], row=id/nq, q=id-row*nq;
  long long expected=counts[id];
  if(expected<min_assets || expected<=0 || expected>2147483647LL) return;
  unsigned long long pos[68]={0}, neg[68]={0};
  long long seen=0;
  for(long long j=0;j<ncols;j++) {
    if(bucket[row*ncols+j]!=q) continue;
    double v=data[row*ncols+j];
    if(!isfinite(v)) continue;
    seen++;
    unsigned long long bits=__double_as_longlong(v);
    unsigned long long mant=bits & 0x000fffffffffffffULL;
    int e=(int)((bits>>52)&0x7ffULL);
    if(e) mant |= 0x0010000000000000ULL;
    if(mant==0) continue;
    int shift=e ? e-1 : 0;
    int limb=shift>>5, off=shift&31;
    unsigned long long lo=mant & 0xffffffffULL;
    unsigned long long hi=mant>>32;
    unsigned long long w0=(lo<<off)&0xffffffffULL;
    unsigned long long c0=off ? (lo>>(32-off)) : 0ULL;
    unsigned long long w1=((hi<<off)|c0)&0xffffffffULL;
    unsigned long long c1=off ? (hi>>(32-off)) : 0ULL;
    unsigned long long* a=(bits>>63) ? neg : pos;
    a[limb]+=w0; if(limb+1<68) a[limb+1]+=w1;
    if(c1 && limb+2<68) a[limb+2]+=c1;
  }
  if(seen!=expected) { atomicExch(error_flag,1); return; }
  for(int i=0;i<67;i++) {
    unsigned long long c=pos[i]>>32; pos[i]&=0xffffffffULL; pos[i+1]+=c;
    c=neg[i]>>32; neg[i]&=0xffffffffULL; neg[i+1]+=c;
  }
  int cmp=0;
  for(int i=67;i>=0;i--) { if(pos[i]>neg[i]){cmp=1;break;} if(pos[i]<neg[i]){cmp=-1;break;} }
  if(cmp==0) { means[id]=0.0; return; }
  unsigned long long mag[68]; unsigned long long borrow=0;
  for(int i=0;i<68;i++) {
    unsigned long long a=cmp>0?pos[i]:neg[i], b=cmp>0?neg[i]:pos[i];
    unsigned long long sub=b+borrow;
    unsigned long long next=(a<sub);
    mag[i]=(a-sub)&0xffffffffULL; borrow=next;
  }
  unsigned long long rem=0;
  for(int i=67;i>=0;i--) {
    unsigned long long cur=(rem<<32)|mag[i];
    mag[i]=cur/(unsigned long long)expected; rem=cur%(unsigned long long)expected;
  }
  int h=-1;
  for(int i=67;i>=0 && h<0;i--) if(mag[i]) {
    unsigned long long z=mag[i]; int b=0; while(z>>1){z>>=1;b++;} h=i*32+b;
  }
  unsigned long long outbits=0;
  if(h<52) {
    unsigned long long sig=0;
    for(int b=0;b<52;b++) if((mag[b>>5]>>(b&31))&1ULL) sig|=1ULL<<b;
    unsigned long long twice=rem*2ULL;
    if(twice>(unsigned long long)expected || (twice==(unsigned long long)expected && (sig&1ULL))) sig++;
    if(sig>=(1ULL<<52)) outbits=1ULL<<52; else outbits=sig;
  } else {
    int drop=h-52; unsigned long long sig=0;
    for(int b=0;b<53;b++) { int src=drop+b; if((mag[src>>5]>>(src&31))&1ULL) sig|=1ULL<<b; }
    int guard=0, sticky=(rem!=0);
    if(drop==0) {
      unsigned long long twice=rem*2ULL;
      if(twice>(unsigned long long)expected || (twice==(unsigned long long)expected && (sig&1ULL))) sig++;
    } else {
      int gb=drop-1; guard=(int)((mag[gb>>5]>>(gb&31))&1ULL);
      for(int b=0;b<gb;b++) if((mag[b>>5]>>(b&31))&1ULL) {sticky=1;break;}
      if(guard && (sticky || (sig&1ULL))) sig++;
    }
    if(sig==(1ULL<<53)){sig>>=1;h++;}
    int e=h-51;
    if(e>=2047) outbits=0x7ff0000000000000ULL;
    else outbits=((unsigned long long)e<<52)|(sig&0x000fffffffffffffULL);
  }
  if(cmp<0) outbits|=0x8000000000000000ULL;
  means[id]=__longlong_as_double((long long)outbits);
}
"""

_RISK_GUARD = r"""
extern "C" __global__ void risk_guard(const int* bucket, const double* data,
    const long long* counts, const double* means, unsigned char* risk,
    int* error_flag, long long nrows,
    long long ncols, long long nq, long long min_assets) {
  long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=nrows) return;
  double mx=0.0;
  for(long long j=0;j<ncols;j++) {
    int b=bucket[row*ncols+j];
    if(b < -1 || b >= nq) atomicExch(error_flag,1);
    double v=data[row*ncols+j];
    if(isfinite(v)) { double a=fabs(v); if(a>mx) mx=a; }
  }
  double scale=mx>0.0?mx:1.0, scaled_sum=0.0;
  if(mx>0.0) for(long long j=0;j<ncols;j++) {
    double v=data[row*ncols+j]; if(isfinite(v)) scaled_sum+=fabs(v)/scale;
  }
  double eps=2.2204460492503130808472633361816e-16;
  double tiny=2.2250738585072013830902327173324e-308;
  double neps=(double)ncols*eps;
  double gamma=neps/(1.0-neps);
  int overflow=(mx>0.0 && mx>1.797693134862315708145274237317e308/fmax(scaled_sum,1.0));
  int subnormal=(mx>0.0 && mx<tiny);
  for(long long q=0;q<nq;q++) {
    long long id=row*nq+q, c=counts[id]; int bad=0;
    if(c<0 || c>ncols) { atomicExch(error_flag,1); risk[id]=0; continue; }
    if(c>=min_assets && c>0) {
      double m=means[id];
      int mean_subnormal=(isfinite(m) && m!=0.0 && fabs(m)<tiny);
      if(!isfinite(m)) bad=1;
      if(overflow || subnormal) bad=1;
      if(isfinite(m) && mx>0.0) {
        double err=2.0*gamma*scaled_sum/(double)c;
        if(err>0.0 && mx>1.0e-12/err) bad=1;
        if(!isfinite(err)) bad=1;
        if(mean_subnormal) bad=1;
      }
      // The row bound above intentionally uses the full reduction width N.
      // For finance-scale rows it can be loose when a bucket contains only a
      // small share of the finite labels. Refine only ordinary finite means;
      // all legacy mixed-scale, overflow, and subnormal risks stay exact.
      if(bad && isfinite(m) && !overflow && !subnormal && !mean_subnormal && mx<=1.0) {
        double bucket_mx=0.0, bucket_scaled_sum=0.0;
        long long seen=0;
        for(long long j=0;j<ncols;j++) {
          if(bucket[row*ncols+j]!=q) continue;
          double v=data[row*ncols+j];
          if(!isfinite(v)) continue;
          seen++;
          double a=fabs(v);
          if(a>bucket_mx) bucket_mx=a;
        }
        if(seen!=c) {
          atomicExch(error_flag,1);
        } else if(bucket_mx>0.0 && bucket_mx<tiny) {
          // A bucket made entirely of subnormals retains the prior exact path.
        } else if(bucket_mx>0.0) {
          for(long long j=0;j<ncols;j++) {
            if(bucket[row*ncols+j]!=q) continue;
            double v=data[row*ncols+j];
            if(isfinite(v)) bucket_scaled_sum+=fabs(v)/bucket_mx;
          }
          double bucket_err=2.0*gamma*bucket_mx*bucket_scaled_sum/(double)c;
          if(isfinite(bucket_err) && bucket_err<=1.0e-12) bad=0;
        } else {
          // A zero-valued bucket has no summation error.
          bad=0;
        }
      }
    }
    risk[id]=(unsigned char)bad;
  }
}
"""


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - host without CUDA
        raise RuntimeError("CuPy is required for GPU quantile-mean repair") from exc
    return cp


def _local_size_bytes(kernel: Any) -> int:
    """Compile a RawKernel and read its per-thread local-memory attribute."""
    kernel.compile()
    attrs = kernel.attributes
    value = attrs.get("local_size_bytes", attrs.get("localSizeBytes"))
    if value is None:
        raise RuntimeError("CuPy did not report RawKernel local_size_bytes")
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise RuntimeError("CuPy reported invalid RawKernel local_size_bytes")
    return int(value)


def repair_quantile_means_gpu(
    bucket: Any,
    labels: Any,
    means: Any,
    counts: Any,
    min_assets: int,
    *,
    workspace_bytes: int,
) -> Any:
    """Repair unsafe finite bucket means in place, returning ``means``.

    Shapes are ``(R,N)``, ``(R,N)``, and ``(R,Q)``. ``bucket`` is int32 with
    ``-1`` for unassigned observations; ``labels`` contains float64 returns.
    Non-finite returns are omitted and must agree with ``counts``. Counts above int32
    are rejected because the fixed-point accumulator's proven bound uses that
    admission limit. Only small risk-count metadata is synchronized to host.

    This kernel is experimental until qualified on supported CUDA devices.
    """
    cp = _cupy()
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")
    if isinstance(min_assets, bool) or not isinstance(min_assets, int) or min_assets < 1:
        raise ValueError("min_assets must be a positive integer")
    if getattr(bucket, "ndim", None) != 2 or getattr(labels, "shape", None) != bucket.shape:
        raise ValueError("bucket and labels must have equal (R,N) shapes")
    if getattr(means, "ndim", None) != 2 or getattr(counts, "shape", None) != means.shape:
        raise ValueError("means and counts must have equal (R,Q) shapes")
    rows, ncols = map(int, bucket.shape)
    q = int(means.shape[1])
    if int(means.shape[0]) != rows or q <= 0:
        raise ValueError("means must have shape (R,Q) with Q > 0")
    if ncols > 2147483647:
        raise ValueError("N exceeds the int32max admission bound")
    if not all(isinstance(a, cp.ndarray) for a in (bucket, labels, means, counts)):
        raise TypeError("all inputs must be CuPy device arrays; CPU fallback is unsupported")
    if bucket.dtype != cp.int32 or labels.dtype != cp.float64 or means.dtype != cp.float64:
        raise TypeError("bucket must be int32 and labels/means float64")
    if counts.dtype != cp.int64:
        raise TypeError("counts must be int64 for the RawKernel ABI")
    arrays = (bucket, labels, means, counts)
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError("all inputs must be C-contiguous")
    device_ids = {a.device.id for a in arrays}
    current_device = cp.cuda.Device().id
    if device_ids != {current_device}:
        raise ValueError("all inputs must be on the current CUDA device")
    means_start = int(means.data.ptr)
    means_stop = means_start + int(means.nbytes)
    for name, source in (("bucket", bucket), ("labels", labels), ("counts", counts)):
        start = int(source.data.ptr)
        stop = start + int(source.nbytes)
        if means_start < stop and start < means_stop:
            raise ValueError(f"means output storage must not overlap {name}")
    if rows == 0:
        return means

    # Normal guard/nonzero working set: risk map + IDs + CuPy nonzero scratch,
    # row scalars, mismatch metadata and actual guard local memory.
    base_required = rows * (34 + 40 * q) + 4
    if base_required > workspace_bytes:
        raise MemoryError(f"quantile guard requires {base_required} workspace bytes")
    guard = cp.RawKernel(_RISK_GUARD, "risk_guard", options=("--std=c++11",))
    guard_local = _local_size_bytes(guard)
    guard_threads = ((rows + 127) // 128) * 128
    normal_required = base_required + guard_local * guard_threads
    if normal_required > workspace_bytes:
        raise MemoryError(f"quantile guard requires {normal_required} workspace bytes")

    risk = cp.empty((rows * q,), dtype=cp.uint8)
    error_flag = cp.zeros((), dtype=cp.int32)
    guard(((rows + 127) // 128,), (128,),
          (bucket, labels, counts, means, risk, error_flag, rows, ncols, q,
           int(min_assets)))
    if int(error_flag.item()):
        raise ValueError("bucket IDs or counts are outside their admitted domains")
    ids = cp.nonzero(risk)[0].astype(cp.int64, copy=False)
    risk_count = int(ids.size)  # allowed synchronization of risk-count metadata
    if risk_count == 0:
        return means
    kernel = cp.RawKernel(_FIXED_MEAN, "fixed_mean", options=("--std=c++11",))
    fixed_local = _local_size_bytes(kernel)
    fixed_required = normal_required + _THREADS * max(8192, fixed_local)
    if fixed_required > workspace_bytes:
        raise MemoryError(f"quantile exact repair requires {fixed_required} workspace bytes")
    for start in range(0, risk_count, _THREADS):
        stop = min(start + _THREADS, risk_count)
        batch_ids = ids[start:stop]
        kernel((1,), (_THREADS,),
               (bucket, labels, means, counts, batch_ids, error_flag, rows,
                ncols, q, int(min_assets), stop - start))
    if int(error_flag.item()):
        raise ValueError("risky bucket count does not match finite selected observations")
    return means


__all__ = ["repair_quantile_means_gpu"]
