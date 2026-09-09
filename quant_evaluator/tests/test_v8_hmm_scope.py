import numpy as np
import pytest
from quant_evaluator.metrics.stats.regime_detection import GaussianHMM

def test_shape_missing_axis_and_refit_reset():
    with pytest.raises(ValueError): GaussianHMM().fit(np.ones((10,2)))
    x=np.r_[np.linspace(-2,-1,10),np.linspace(1,2,10)]
    with pytest.raises(ValueError): GaussianHMM(n_iter=20).fit(np.insert(x,10,np.nan))
    m=GaussianHMM(n_iter=20,random_state=1).fit(x)
    p=m.predict_proba(x,"filtered_asof")
    assert p.shape==(20,2)
    m.n_iter=1; m.fit(x)
    assert m.n_iter_fit_==1 and not m.converged_

def test_filtered_prefix_invariance_and_smoothed_label():
    x=np.r_[np.linspace(-1,-.1,20),np.linspace(.1,1,20)]
    m=GaussianHMM(n_iter=30,random_state=2).fit(x)
    assert np.allclose(m.predict_proba(x[:25],"filtered_asof"),m.predict_proba(x,"filtered_asof")[:25])
    assert m.predict_proba(x,"smoothed_posthoc").shape==(40,2)


@pytest.mark.parametrize("x", [np.array([]), np.array([1.0])], ids=["empty", "one-sample"])
def test_t122_empty_and_one_sample_reject_at_sequence_boundary(x):
    with pytest.raises(ValueError, match="Insufficient data"):
        GaussianHMM().fit(x)


@pytest.mark.parametrize("n_iter", [0, -1, 1.5, True])
def test_t126_invalid_n_iter_rejected(n_iter):
    with pytest.raises(ValueError, match="n_iter"):
        GaussianHMM(n_iter=n_iter)


@pytest.mark.parametrize("tol", [0.0, -0.1, np.nan, np.inf, True])
def test_t126_invalid_tol_rejected(tol):
    with pytest.raises(ValueError, match="tol"):
        GaussianHMM(tol=tol)


def test_t126_empty_transition_row_invalidates_fit(monkeypatch):
    model = GaussianHMM(n_iter=2, random_state=1)
    original_update = model._update_params

    def inject_empty_row(gamma, xi, x):
        original_update(gamma, xi, x)
        model.trans_mat_[0] = 0.0

    monkeypatch.setattr(model, "_update_params", inject_empty_row)
    with pytest.raises(FloatingPointError, match="invalid transition matrix"):
        model.fit(np.linspace(-1.0, 1.0, 20))
    assert model.n_iter_fit_ == 0
    assert model.trans_mat_ is None


def test_t129_state_parameter_and_weight_permutation_is_output_invariant():
    x = np.r_[np.linspace(-2.0, -0.5, 20), np.linspace(0.5, 2.0, 20)]
    model = GaussianHMM(n_iter=30, random_state=4).fit(x)
    probabilities = model.predict_proba(x, "filtered_asof")
    economic_weights = np.array([-0.75, 1.25])

    permutation = np.array([1, 0])
    permuted = GaussianHMM(n_states=2)
    permuted.start_prob_ = model.start_prob_[permutation]
    permuted.trans_mat_ = model.trans_mat_[np.ix_(permutation, permutation)]
    permuted.means_ = model.means_[permutation]
    permuted.vars_ = model.vars_[permutation]
    permuted.n_iter_fit_ = model.n_iter_fit_
    permuted_probabilities = permuted.predict_proba(x, "filtered_asof")

    np.testing.assert_allclose(permuted_probabilities, probabilities[:, permutation])
    np.testing.assert_allclose(
        permuted_probabilities @ economic_weights[permutation],
        probabilities @ economic_weights,
    )


def test_t127_smoothed_prefix_changes_with_future_tail_but_filtered_does_not():
    train=np.r_[np.linspace(-2.,-.2,30),np.linspace(.2,2.,30)]
    model=GaussianHMM(n_iter=40,random_state=9).fit(train)
    prefix=np.linspace(-.3,.3,20)
    low=np.r_[prefix,np.full(20,-2.)]
    high=np.r_[prefix,np.full(20,2.)]
    np.testing.assert_allclose(model.predict_proba(low,'filtered_asof')[:20],
                               model.predict_proba(high,'filtered_asof')[:20])
    assert not np.allclose(model.predict_proba(low,'smoothed_posthoc')[:20],
                           model.predict_proba(high,'smoothed_posthoc')[:20])
