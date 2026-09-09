import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import research_spectral as rs


def original_matrix_reference(x, y, z, purge):
    """Historical zero-intercept M34 oracle, not the M35 model contract."""
    n = len(x)
    split = n // 2
    rx, ry = np.full(n, np.nan), np.full(n, np.nan)
    folds = ((np.arange(split), np.arange(split+purge,n)),
             (np.arange(split,n), np.arange(split-purge)))
    for tr, te in folds:
        sigma = np.median(np.abs(z[tr,None]-z[None,tr]))
        if not np.isfinite(sigma) or sigma <= rs._EPS:
            sigma = 1.
        kz = rs._rbf(z[tr,None],z[tr,None],sigma)
        kt = rs._rbf(z[te,None],z[tr,None],sigma)
        lam = 1e-3*np.trace(kz)/len(tr)
        smooth = kt @ np.linalg.solve(kz+lam*np.eye(len(tr)),np.eye(len(tr)))
        rx[te], ry[te] = x[te]-smooth@x[tr], y[te]-smooth@y[tr]
    mask = np.isfinite(rx)&np.isfinite(ry)
    rx, ry = rx[mask], ry[mask]
    kernels=[]
    for a in (rx,ry):
        sigma=np.median(np.abs(a[:,None]-a[None,:]))
        if not np.isfinite(sigma) or sigma <= rs._EPS:
            sigma=1.
        kernels.append(rs._rbf(a[:,None],a[:,None],sigma))
    h=np.eye(len(rx))-np.ones((len(rx),len(rx)))/len(rx)
    return np.trace(kernels[0]@h@kernels[1]@h)/(len(rx)-1)**2


def intercept_kkt_reference(x, y, z, purge):
    """Independent augmented-system fit; does not call centering/fit helpers."""
    n, split = len(x), len(x)//2
    residual = np.full((n, 2), np.nan)
    response = np.column_stack((x, y))
    for tr, te in ((np.arange(split), np.arange(split+purge, n)),
                   (np.arange(split, n), np.arange(split-purge))):
        distance = z[tr, None] - z[None, tr]
        sigma = np.median(np.abs(distance))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        gram = np.exp(-distance**2/(2*sigma**2))
        cross = np.exp(-(z[te, None]-z[None, tr])**2/(2*sigma**2))
        h = np.eye(len(tr))-np.ones((len(tr), len(tr)))/len(tr)
        trace = np.trace(h @ gram @ h)
        lam = 1e-3*(trace/len(tr) if trace > 1e-12 else 1.0)
        augmented = np.block([[gram+lam*np.eye(len(tr)), np.ones((len(tr), 1))],
                              [np.ones((1, len(tr))), np.zeros((1, 1))]])
        coefficients = np.linalg.solve(augmented, np.vstack([response[tr], np.zeros((1, 2))]))
        residual[te] = response[te] - cross @ coefficients[:-1] - coefficients[-1]
    residual = residual[np.isfinite(residual).all(axis=1)]
    kernels = []
    for a in residual.T:
        distance = a[:, None]-a[None, :]
        sigma = np.median(np.abs(distance))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        kernels.append(np.exp(-distance**2/(2*sigma**2)))
    h = np.eye(len(residual))-np.ones((len(residual), len(residual)))/len(residual)
    return np.trace(kernels[0] @ h @ kernels[1] @ h)/(len(residual)-1)**2


@pytest.mark.parametrize("n,purge", [(24,6),(31,3),(60,3)])
@pytest.mark.parametrize("kind", ["random","constant","duplicate"])
def test_rhs_is_two_columns_and_value_matches_declared_intercept_model(n,purge,kind,monkeypatch):
    rng=np.random.default_rng(934)
    x,y,z=rng.normal(size=(3,n))
    if kind=="constant": x=np.ones(n)
    if kind=="duplicate": y=x.copy()
    reference=intercept_kkt_reference(x,y,z,purge)
    solve=np.linalg.solve
    shapes=[]
    def recorded(a,b):
        shapes.append(b.shape)
        return solve(a,b)
    monkeypatch.setattr(np.linalg,"solve",recorded)
    actual=rs._residualized_hsic(x,y,z,purge)
    assert shapes==[(n//2,2),(n-n//2,2)]
    assert actual==pytest.approx(reference,rel=1e-8,abs=1e-11)


@pytest.mark.parametrize("window,purge", [(23,3),(30,10),(31,10)])
def test_impossible_purge_domain_is_rejected_before_kernel(window,purge,monkeypatch):
    load_all()
    op=OperatorRegistry.get("ts_residualized_hsic",backend="pandas_numpy",mode="research")
    def forbidden(*args,**kwargs): pytest.fail("impossible fold reached kernel")
    monkeypatch.setattr(op,"_calculate_series",forbidden)
    frame=pd.DataFrame({"A":np.arange(50.)})
    with pytest.raises(ValueError):
        op.calculate(frame,frame,frame,window=window,purge_gap=purge)
