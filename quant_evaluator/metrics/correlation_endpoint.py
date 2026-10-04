"""Certify exact affine correlation endpoints without numerical clipping."""
from fractions import Fraction
import numpy as np


def _stored_scalar_fraction(value):
    """Represent the scalar exactly, without a float64 coercion."""
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def certify_correlation_endpoint(value, x, y):
    """Return exact +/-1 only for a proved nonconstant affine relation."""
    # This is a candidate filter, not an equality tolerance. Only an exact
    # proof below can change a value; ordinary rows retain their old bits.
    if not np.isfinite(value) or abs(value) < 1 - 8 * np.finfo(np.float64).eps:
        return value
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        return value
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return value
    if np.all(x == x[0]) or np.all(y == y[0]):
        return value
    # Cross-dtype equality can coerce large integers; integer negation can
    # wrap at fixed width. Neither operation is an exact proof in that case.
    if x.dtype == y.dtype and np.array_equal(x, y):
        return 1.0
    if x.dtype == y.dtype and x.dtype.kind == "f" and np.array_equal(x, -y):
        return -1.0
    # Scalar.as_integer_ratio and integer conversion describe stored binary
    # values exactly. Cross multiplication proves affinity without rounded
    # means, differences, division, or an epsilon that erases real variation.
    xa, ya = _stored_scalar_fraction(x[0]), _stored_scalar_fraction(y[0])
    basis = None
    for a, b in zip(x, y):
        dx, dy = _stored_scalar_fraction(a) - xa, _stored_scalar_fraction(b) - ya
        if not dx and not dy:
            continue
        if not dx or not dy:
            return value
        if basis is None:
            basis = (dx, dy)
        elif dx * basis[1] != dy * basis[0]:
            return value
    if basis is None:
        return value
    return 1.0 if basis[0] * basis[1] > 0 else -1.0


def certify_ic_panel_endpoints(result, values, labels, mask, *, method):
    """Apply the endpoint contract to optional host backends, in place.

    Scratch is one compressed cross-section, never another factor cube.
    Ordinary rows are not read or modified; counts remain owned by caller.
    """
    candidates = np.isfinite(result) & (np.abs(result) >= 1 - 8 * np.finfo(np.float64).eps)
    for t, f in np.argwhere(candidates):
        valid = mask[t, :, f]
        x, y = values[t, valid, f], labels[t, valid]
        if method == "spearman":
            from scipy.stats import rankdata
            x, y = rankdata(x, method="average"), rankdata(y, method="average")
        result[t, f] = certify_correlation_endpoint(result[t, f], x, y)
    return result
