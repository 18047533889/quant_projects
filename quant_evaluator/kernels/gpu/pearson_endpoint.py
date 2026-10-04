"""Exact Pearson endpoint certification with candidate-row-only host proofs."""

from __future__ import annotations

import numpy as np

from quant_evaluator.metrics.correlation_endpoint import certify_correlation_endpoint


def certify_gpu_pearson_endpoints(ic, x, y, finite, *, chunk_rows: int = 64):
    """Prove near-endpoint rows exactly, copying at most ``chunk_rows`` to host.

    Candidate indices and equality flags cross to host for control flow.
    Candidate discovery and gathering stay on device. Only unresolved
    candidate vectors cross the device boundary; chunking bounds temporary
    memory even when many rows are exact endpoints.
    """
    import cupy as cp

    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or not 1 <= chunk_rows <= 64:
        raise ValueError("chunk_rows must be an integer from 1 through 64")
    near = cp.isfinite(ic) & (cp.abs(ic) >= 1.0 - 8.0 * np.finfo(np.float64).eps)
    positions = cp.argwhere(near)
    pos_host = cp.asnumpy(positions)
    if pos_host.size == 0:
        return ic

    out = ic.copy()
    for start in range(0, len(pos_host), chunk_rows):
        pos = pos_host[start:start + chunk_rows]
        ti = cp.asarray(pos[:, 0], dtype=cp.int64)
        fi = cp.asarray(pos[:, 1], dtype=cp.int64)
        yi = fi if y.shape[1] != 1 else cp.zeros_like(fi)
        xc = x[ti, fi, :]
        yc = y[ti, yi, :]
        fc = finite[ti, fi, :]
        identical = cp.all((~fc) | (xc == yc), axis=1)
        opposite = cp.all((~fc) | (xc == -yc), axis=1)
        fast_host = cp.asnumpy(cp.stack((identical, opposite), axis=1))
        if fast_host.any():
            same = cp.asarray(fast_host[:, 0])
            opp = cp.asarray(fast_host[:, 1])
            out[ti[same], fi[same]] = 1.0
            out[ti[opp], fi[opp]] = -1.0
        unresolved_host = ~(fast_host[:, 0] | fast_host[:, 1])
        if unresolved_host.any():
            keep = cp.asarray(unresolved_host)
            xv = cp.asnumpy(xc[keep])
            yv = cp.asnumpy(yc[keep])
            fv = cp.asnumpy(fc[keep])
            values = cp.asnumpy(ic[ti[keep], fi[keep]]).copy()
            for j in range(len(values)):
                values[j] = certify_correlation_endpoint(values[j], xv[j, fv[j]], yv[j, fv[j]])
            out[ti[keep], fi[keep]] = cp.asarray(values, dtype=ic.dtype)
    return out
