"""X01 actual-library counterexamples, independent of the old panel mean."""
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.metrics.exposure_evidence import ExposurePanel, compute_exposure_evidence
from quant_evaluator.tests.test_public_gpu_exposure_v3 import _known_math_contract
from quant_evaluator.runtime.evaluator import evaluate


def orthogonal_fixture():
    n=40
    z=np.linspace(-1,1,n); z-=z.mean(); z/=np.std(z)
    a=z*z; a-=a.mean(); a/=np.std(a)
    risk=np.tile(z[None,:,None],(8,1,1))
    return risk,np.tile(a[None,:],(8,1))


def test_x01_direct_bundle_factor_equals_centered_size_is_fully_explained():
    risk,_=orthogonal_fixture()
    panel=ExposurePanel(risk,style_names=("size",),source_ref="risk:synthetic")
    result=compute_exposure_evidence(panel,risk[:,:,0],risk[:,:,0])
    assert result["size_exposure"]==pytest.approx(1.,abs=1e-12)
    assert result["purity_ratio"]==pytest.approx(0.,abs=1e-12)


@pytest.mark.parametrize("backend",["cpu","cuda_strict"])
def test_x01_public_purity_is_residual_fraction_not_loading_dispersion(backend):
    risk,alpha=orthogonal_fixture()
    factor=risk[:,:,0]+alpha
    batch,labels,panel=_known_math_contract(risk,factor,("size",))
    out=evaluate(batch,labels,metrics=("size_exposure","purity_ratio"),exposure_panel=panel,backend=backend)
    assert out.get_metric("purity_ratio","known").value==pytest.approx(.5,abs=1e-12)
    assert out.get_metric("size_exposure","known").value==pytest.approx(1/np.sqrt(2),abs=1e-12)
    shifted=replace(panel,values=panel.values*1000+1234567)
    out2=evaluate(batch,labels,metrics=("size_exposure","purity_ratio"),exposure_panel=shifted,backend=backend)
    for mid in ("size_exposure","purity_ratio"):
        assert out2.get_metric(mid,"known").value==pytest.approx(out.get_metric(mid,"known").value,abs=1e-10)


def test_x01_constant_factor_never_has_perfect_purity():
    risk,_=orthogonal_fixture()
    batch,labels,panel=_known_math_contract(risk,np.ones(risk.shape[:2]),("size",))
    result=evaluate(batch,labels,metrics=("size_exposure","purity_ratio"),exposure_panel=panel)
    assert not result.get_metric("purity_ratio","known").valid
    assert not result.get_metric("size_exposure","known").valid


def test_x09_ill_conditioned_coefficients_do_not_erase_real_residual():
    from quant_evaluator.metrics.exposure import rank_aware_projection
    n=40
    basis=np.linalg.qr(np.column_stack((np.ones(n),np.linspace(-1,1,n),
        np.linspace(-1,1,n)**2,np.linspace(-1,1,n)**3)))[0]
    z,u,y=basis[:,1],basis[:,2],basis[:,3]
    _,residual,diagnostics=rank_aware_projection(np.column_stack((z,z+1e-12*u)),y)
    assert diagnostics["condition"]>1e10
    assert np.linalg.norm(residual)>.99
    assert diagnostics["status"]!="NO_RESIDUAL_VARIANCE"


@pytest.mark.parametrize("backend",["cpu","cuda_strict"])
def test_weighted_public_series_provenance_and_roundtrip(backend):
    from quant_evaluator.contracts._ndarray_codec import decode_value
    risk,alpha=orthogonal_fixture()
    batch,labels,panel=_known_math_contract(risk,risk[:,:,0]+alpha,("size",))
    weights=np.tile(np.linspace(.1,2,risk.shape[1]),(risk.shape[0],1))
    panel=replace(panel,regression_weights=weights,weight_ref="weights:gold")
    panel=ExposurePanel.from_dict(panel.to_dict())
    result=evaluate(batch,labels,metrics=("size_exposure","purity_ratio"),exposure_panel=panel,backend=backend)
    artifact=result.artifacts["purity_ratio"]
    series=artifact.provenance["factor_loading_series"][0]
    z=risk[0,:,0]; y=batch.values[0,:,0]; w=weights[0]/weights[0].sum()
    zc=z-np.sum(w*z); yc=y-np.sum(w*y)
    beta=np.sum(w*zc*yc)/np.sum(w*zc*zc)
    purity=np.sum(w*(yc-beta*zc)**2)/np.sum(w*yc*yc)
    assert result.get_metric("purity_ratio","known").value==pytest.approx(purity,abs=1e-12)
    assert np.asarray(decode_value(series["r_squared"]))==pytest.approx(np.full(8,1-purity))
    assert artifact.provenance["weight_ref"]=="weights:gold"
    assert artifact.provenance["observation_counts"]==tuple([8])
    assert artifact.producer_version=="4.0.0"


def test_reused_evaluator_binds_risk_weights_and_preserves_series_on_cache_hit():
    from quant_evaluator.runtime.evaluator import Evaluator
    risk,alpha=orthogonal_fixture()
    batch,labels,panel=_known_math_contract(risk,risk[:,:,0]+alpha,("size",))
    runtime=Evaluator()
    kwargs=dict(metrics=("size_exposure","purity_ratio"),evaluator=runtime)
    original=evaluate(batch,labels,exposure_panel=panel,**kwargs)
    repeated=evaluate(batch,labels,exposure_panel=panel,**kwargs)
    assert original.get_metric("purity_ratio","known").value==repeated.get_metric("purity_ratio","known").value
    weights=np.tile(np.linspace(.1,2,risk.shape[1]),(risk.shape[0],1))
    changed=replace(panel,regression_weights=weights,weight_ref="weights:changed")
    weighted=evaluate(batch,labels,exposure_panel=changed,**kwargs)
    assert weighted.get_metric("purity_ratio","known").value!=pytest.approx(original.get_metric("purity_ratio","known").value)
    assert weighted.config_hash!=original.config_hash


def test_security_panel_nanosecond_date_axis_json_roundtrip():
    import json
    risk,_=orthogonal_fixture()
    dates=np.arange(8).astype("datetime64[ns]")
    panel=ExposurePanel(risk,style_names=("size",),date_index=tuple(dates))
    restored=ExposurePanel.from_dict(json.loads(json.dumps(panel.to_dict())))
    np.testing.assert_array_equal(restored.date_index,panel.date_index)
    assert np.asarray(restored.date_index).dtype==np.dtype("datetime64[ns]")


@pytest.mark.parametrize("backend",["cpu","cuda_strict"])
@pytest.mark.parametrize("parameters",[{"min_finite":True},{"min_finite":0},{"min_obs":1},{"weights":[1]}])
def test_public_exposure_parameter_validation_precedes_backend_dispatch(backend,parameters):
    from quant_evaluator.runtime.evaluator import InvalidContractError
    risk,alpha=orthogonal_fixture()
    batch,labels,panel=_known_math_contract(risk,risk[:,:,0]+alpha,("size",))
    with pytest.raises(InvalidContractError):
        evaluate(batch,labels,metrics=("purity_ratio",),exposure_panel=panel,backend=backend,
                 metric_parameters={"purity_ratio":parameters})
