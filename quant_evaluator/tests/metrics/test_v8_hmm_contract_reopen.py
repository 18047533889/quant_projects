import numpy as np
import pytest
from quant_evaluator.metrics.stats.regime_detection import GaussianHMM,RegimeInference,require_production_inference

def test_viterbi_compatibility_and_typed_scopes():
    x=np.r_[np.linspace(-2,-1,20),np.linspace(1,2,20)]
    model=GaussianHMM(n_iter=30,random_state=1).fit(x)
    assert model.predict(x).shape==(40,)
    filtered=model.infer(x,scope='FILTERED_ASOF',fit_end=20,decision_time=40)
    smooth=model.infer(x,scope='SMOOTHED_POSTHOC',fit_end=20,decision_time=40)
    assert isinstance(filtered,RegimeInference) and require_production_inference(filtered) is filtered
    with pytest.raises(ValueError): require_production_inference(smooth)
    assert np.allclose(model.predict_proba(x),smooth.probabilities)
    with pytest.raises(ValueError): model.infer(x,scope='FILTERED_ASOF')
    with pytest.raises(ValueError): model.infer(x,scope='FILTERED_ASOF',fit_end=2,decision_time=1)

def test_failed_refit_invalidates_old_model_and_gap_requires_clock_policy():
    x=np.linspace(-1,1,40); model=GaussianHMM(n_iter=20).fit(x)
    with pytest.raises(ValueError): model.fit(np.array([1.,np.nan,2.,3.]))
    with pytest.raises(RuntimeError): model.predict(np.array([1.,2.]))
    with pytest.raises(ValueError): GaussianHMM(missing_policy='compress_valid')

def test_likelihood_decrease_is_failure_not_convergence(monkeypatch):
    x=np.linspace(-1,1,40); model=GaussianHMM(n_iter=5)
    original=model._forward; calls={'n':0}
    def decreasing(emission):
        alpha,ll=original(emission); calls['n']+=1
        return alpha,ll-(100 if calls['n']>1 else 0)
    monkeypatch.setattr(model,'_forward',decreasing)
    with pytest.raises(FloatingPointError): model.fit(x)
    assert not model.converged_ and model.n_iter_fit_==0
