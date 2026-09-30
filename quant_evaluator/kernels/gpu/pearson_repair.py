"""Device-side selective repair for unsafe float64 Pearson rows."""
from __future__ import annotations

_KERNEL = r'''
extern "C" __global__
void repair(const double* x, const double* y, const unsigned char* finite,
            const unsigned char* unsafe, double* out, int n, int nf,
            int y_broadcast) {
    int row = (int)blockIdx.x;
    if (!unsafe[row]) return;
    int lane=(int)threadIdx.x;
    __shared__ double a[128], b[128], cs[128];
    __shared__ int q[128], first_shared;
    int yr = y_broadcast ? row / nf : row;
    const double* xr = x + ((long long)row) * n;
    const double* yp = y + ((long long)yr) * n;
    const unsigned char* fr = finite + ((long long)row) * n;
    int first=-1;
    if(lane==0) for(int i=0;i<n;++i) if(fr[i]) { first=i; break; }
    if(lane==0) first_shared=first;
    __syncthreads();
    first=first_shared;
    if(first<0) { if(lane==0) out[row]=nan(""); return; }
    double ox=xr[first], oy=yp[first];
    double lx=0.0,ly=0.0;
    for(int i=lane;i<n;i+=128) if(fr[i]) { lx=fmax(lx,fabs(xr[i]));ly=fmax(ly,fabs(yp[i])); }
    a[lane]=lx;b[lane]=ly;__syncthreads();
    for(int d=64;d>0;d>>=1) { if(lane<d){a[lane]=fmax(a[lane],a[lane+d]);b[lane]=fmax(b[lane],b[lane+d]);} __syncthreads(); }
    double maxx=fmax(a[0],1.0),maxy=fmax(b[0],1.0);
    int overx=0,overy=0;
    for(int i=lane;i<n;i+=128) if(fr[i]) { overx|=!isfinite(xr[i]-ox);overy|=!isfinite(yp[i]-oy); }
    q[lane]=overx;__syncthreads();
    for(int d=64;d>0;d>>=1){if(lane<d)q[lane]|=q[lane+d];__syncthreads();}
    overx=q[0];
    // All warps must capture the result before any lane reuses q.
    __syncthreads();
    q[lane]=overy;__syncthreads();
    for(int d=64;d>0;d>>=1){if(lane<d)q[lane]|=q[lane+d];__syncthreads();}
    overy=q[0];
    double lxbase=0.0,lybase=0.0;
    for(int i=lane;i<n;i+=128) if(fr[i]) {
        double u=overx?xr[i]/maxx:xr[i]-ox,v=overy?yp[i]/maxy:yp[i]-oy;
        lxbase=fmax(lxbase,fabs(u));lybase=fmax(lybase,fabs(v));
    }
    a[lane]=lxbase;b[lane]=lybase;__syncthreads();
    for(int d=64;d>0;d>>=1){if(lane<d){a[lane]=fmax(a[lane],a[lane+d]);b[lane]=fmax(b[lane],b[lane+d]);}__syncthreads();}
    double bxmax=a[0],bymax=b[0];
    __syncthreads();
    if(bxmax==0.0||bymax==0.0){if(lane==0)out[row]=nan("");return;}
    double sx=0.0,sy=0.0;int cnt=0;
    for(int i=lane;i<n;i+=128)if(fr[i]){
        double u=overx?xr[i]/maxx:xr[i]-ox,v=overy?yp[i]/maxy:yp[i]-oy;
        sx+=u/bxmax;sy+=v/bymax;++cnt;
    }
    a[lane]=sx;b[lane]=sy;q[lane]=cnt;__syncthreads();
    for(int d=64;d>0;d>>=1){if(lane<d){a[lane]+=a[lane+d];b[lane]+=b[lane+d];q[lane]+=q[lane+d];}__syncthreads();}
    double mx=a[0]/q[0],my=b[0]/q[0],vx=0.0,vy=0.0,cov=0.0;
    __syncthreads();
    for(int i=lane;i<n;i+=128)if(fr[i]){
        double u=overx?xr[i]/maxx:xr[i]-ox,v=overy?yp[i]/maxy:yp[i]-oy;
        double dx=u/bxmax-mx,dy=v/bymax-my;vx+=dx*dx;vy+=dy*dy;cov+=dx*dy;
    }
    a[lane]=vx;b[lane]=vy;cs[lane]=cov;__syncthreads();
    for(int d=64;d>0;d>>=1){if(lane<d){a[lane]+=a[lane+d];b[lane]+=b[lane+d];cs[lane]+=cs[lane+d];}__syncthreads();}
    if(lane==0)out[row]=(a[0]>0.0&&b[0]>0.0)?(cs[0]/sqrt(a[0]))/sqrt(b[0]):nan("");
}
'''

def repair_unsafe_pearson_rows(x, y, finite, unsafe, ic, *, n_factors: int, y_is_broadcast: bool):
    import cupy as cp
    if getattr(x, "ndim", None) != 3 or getattr(y, "ndim", None) != 3:
        raise ValueError("repair expects x/y with shape (T,F,N) and (T,1|F,N)")
    expected_output = x.shape[:2]
    if (y.shape[0] != x.shape[0] or y.shape[2] != x.shape[2]
            or y.shape[1] not in (1, x.shape[1])
            or finite.shape != x.shape or unsafe.shape != expected_output
            or ic.shape != expected_output or n_factors != x.shape[1]
            or y_is_broadcast != (y.shape[1] == 1)):
        raise ValueError("unsafe Pearson repair received inconsistent tensor shapes")
    if ic.size == 0:
        return ic
    xd=cp.ascontiguousarray(x,dtype=cp.float64)
    yd=cp.ascontiguousarray(y,dtype=cp.float64)
    fd=cp.ascontiguousarray(finite.reshape(-1,x.shape[-1]),dtype=cp.uint8)
    ud=cp.ascontiguousarray(unsafe.reshape(-1),dtype=cp.uint8)
    out=cp.ascontiguousarray(ic.reshape(-1))
    rows,width=fd.shape
    _kernel()((rows,),(128,),(xd,yd,fd,ud,out,cp.int32(width),
        cp.int32(n_factors),cp.int32(y_is_broadcast)))
    return out.reshape(ic.shape)

def _kernel():
    import cupy as cp
    global _COMPILED_KERNEL
    try: return _COMPILED_KERNEL
    except NameError:
        _COMPILED_KERNEL=cp.RawKernel(_KERNEL,"repair",options=("--std=c++11",))
        return _COMPILED_KERNEL
