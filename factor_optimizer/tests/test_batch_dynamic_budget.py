"""Dynamic TRAIN proposals must not abort unrelated factors."""
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch


def fixture():
    rng = np.random.default_rng(82)
    t, n = 240, 40
    y = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)]) ** 2
    ta = AxisRef('time', 'int', t, np.arange(t))
    aa = AxisRef('asset', 'str', n, np.array([f'a{i}' for i in range(n)]))
    batch = FactorBatch(('good', 'reverse'), ta, aa, np.stack((y, -y), axis=-1))
    labels = LabelBundle('synthetic', y, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t+1)), label_end_time=tuple(range(2, t+2)), asset_axis=aa)
    return batch, labels


@pytest.mark.parametrize('reverse_order', [False, True])
def test_dynamic_budget_overflow_isolates_factor_without_truncating(monkeypatch, reverse_order):
    import factor_optimizer.research_diagnostics as diagnostics
    original = diagnostics.diagnose_training_batch

    def diagnose(batch, labels, **kwargs):
        result = original(batch, labels, **kwargs)
        for name, record in result.items():
            record['proposed_shape_family'] = 'U_SHAPE_REPAIR' if name == 'good' else None
            record['proposed_center'] = .41
        return result

    monkeypatch.setattr(diagnostics, 'diagnose_training_batch', diagnose)
    batch, labels = fixture()
    order = [1, 0] if reverse_order else [0, 1]
    batch = replace(batch, factor_ids=tuple(batch.factor_ids[i] for i in order),
                    values=batch.values[:, :, order])
    config = BatchOptimizationConfig(selection_objective='rank_ic',
        families=('SIGN_ORIENTATION', 'U_SHAPE_REPAIR'), maximum_candidates=7,
        bootstrap_draws=99)
    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    blocked = result.factors['good']
    assert blocked.status == 'budget_exceeded_raw_retained'
    assert blocked.selected_family == 'NO_OP_RAW'
    assert blocked.validation_candidate_identity is None
    assert blocked.candidates == ()
    assert blocked.training_diagnostics['candidate_budget'] == {
        'required': 9, 'maximum': 7, 'status': 'exceeded', 'evaluated': 0}
    np.testing.assert_array_equal(result.optimized.values[:, :, order.index(0)],
                                  batch.values[:, :, order.index(0)])
    assert result.factors['reverse'].selected_family == 'SIGN_ORIENTATION'
    assert result.factors['reverse'].status == 'improved'
    assert result.test_evaluated is False
