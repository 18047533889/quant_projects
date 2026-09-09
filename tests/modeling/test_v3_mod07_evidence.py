import numpy as np
import pytest
from modeling.evaluation import evaluate_predictions, EvaluationContractError, ModelPredictionArtifact
from jobs.evaluate_model_predictions import evaluate_model_predictions
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact


def inputs():
    rng = np.random.default_rng(780)
    times = np.arange('2024-01-01', '2024-02-10', dtype='datetime64[D]')
    assets = np.arange(30)
    x = rng.normal(size=(len(times), len(assets)))
    batch = FactorBatch(('prediction-1',), AxisRef('time', 'datetime64[D]', len(times), times),
                        AxisRef('asset', 'int64', len(assets), assets), x[..., None])
    label = LabelBundle('forward', x * .01, 1, decision_time=tuple(times),
                        label_start_time=tuple(times + 1), label_end_time=tuple(times + 2),
                        asset_axis=batch.asset_axis)
    rows = (x.ravel(), label.values.ravel(), np.repeat(times, len(assets)), np.tile(assets, len(times)))
    return rows, batch, label


def test_existing_public_report_separates_predictions_from_independent_pnl():
    rows, batch, label = inputs()
    returns = np.tile([.01, -.03], 20)[:, None]
    pnl = ProbePortfolioArtifact(returns, time_index=label.decision_time, factor_ids=batch.factor_ids)
    report = evaluate_model_predictions(*rows, prediction_id='prediction-1', prediction_batch=batch,
                                  label_bundle=label, portfolio_returns=pnl)
    assert report.rank_ic == pytest.approx(1)
    assert report.long_short_spread > 0
    assert report.portfolio_evidence['qe_bundle'] != report.prediction_evidence['qe_bundle']
    assert report.portfolio_evidence['execution_certified'] is False
    from quant_evaluator.api.requests import EvaluationBundle
    evidence = EvaluationBundle.from_dict(report.portfolio_evidence['qe_bundle'])
    assert evidence.get_metric('sharpe_ratio', 'prediction-1').value < 0
    assert 'factor_grades' not in report.to_dict()


def test_legacy_label_spread_cannot_be_portfolio_evidence():
    rows, _, _ = inputs()
    report = evaluate_predictions(*rows)
    assert not report.prediction_evidence and not report.portfolio_evidence
    assert report.research_diagnostics['forward_label_spread'] == report.long_short_spread
    assert report.research_diagnostics['oos_certified'] is False
    assert report.portfolio_support['execution_certified'] is False


def test_axis_and_value_misbindings_are_rejected_at_existing_public_entry():
    rows, batch, label = inputs()
    with pytest.raises(EvaluationContractError, match='time/security axes'):
        evaluate_model_predictions(rows[0], rows[1], rows[2], rows[3][::-1],
                             prediction_id='prediction-1', prediction_batch=batch, label_bundle=label)
    with pytest.raises(EvaluationContractError, match='typed QE values'):
        evaluate_model_predictions(-rows[0], *rows[1:], prediction_id='prediction-1',
                             prediction_batch=batch, label_bundle=label)


def test_training_report_caller_builds_qe_batch_from_immutable_prediction_artifact():
    rows, batch, label = inputs()
    artifact = ModelPredictionArtifact(
        prediction_id='prediction-1', values=batch.values[..., 0],
        time_axis=tuple(batch.time_axis.values), asset_axis=tuple(batch.asset_axis.values),
        model_artifact_ref='model:M1', feature_manifest_ref='features:FS1',
        oos_evidence_ref='fold:OOS1',
    )
    report = evaluate_model_predictions(
        prediction_artifact=artifact, label_bundle=label,
        training_importance_evidence={'feature_a': 'train-importance:I1'},
        oos_ablation_evidence_refs={'feature_b': 'oos-ablation:A1'},
    )
    assert report.prediction_id == 'prediction-1'
    assert report.prediction_evidence['qe_bundle']
    assert report.training_importance_evidence == {'feature_a': 'train-importance:I1'}
    assert report.oos_ablation_evidence_refs == {'feature_b': 'oos-ablation:A1'}
    assert 'sharpe_ratio' not in report.training_importance_evidence
    with pytest.raises(ValueError, match='read-only'):
        artifact.values[0, 0] = 0
    with pytest.raises(ValueError, match='WRITEABLE'):
        artifact.values.flags.writeable = True


def test_prediction_artifact_axis_and_evidence_classes_fail_closed():
    rows, batch, label = inputs()
    artifact = ModelPredictionArtifact(
        prediction_id='prediction-1', values=batch.values[..., 0],
        time_axis=tuple(batch.time_axis.values), asset_axis=tuple(batch.asset_axis.values),
        model_artifact_ref='model:M1', feature_manifest_ref='features:FS1',
        oos_evidence_ref='fold:OOS1',
    )
    from dataclasses import replace
    shifted = replace(label, decision_time=tuple(np.asarray(label.decision_time) + np.timedelta64(1, 'D')))
    with pytest.raises(ValueError, match='decision axis'):
        evaluate_model_predictions(prediction_artifact=artifact, label_bundle=shifted)
    with pytest.raises(EvaluationContractError, match='must be separate'):
        evaluate_model_predictions(
            prediction_artifact=artifact, label_bundle=label,
            training_importance_evidence={'feature_a': 'train:I1'},
            oos_ablation_evidence_refs={'feature_a': 'oos:A1'},
        )
    with pytest.raises(EvaluationContractError, match='portfolio metrics are model-level'):
        evaluate_model_predictions(
            prediction_artifact=artifact, label_bundle=label,
            training_importance_evidence={'sharpe_ratio': 'portfolio:P1'},
        )


@pytest.mark.parametrize('field', ['prediction_id', 'model_artifact_ref',
                                   'feature_manifest_ref', 'oos_evidence_ref'])
def test_prediction_artifact_rejects_blank_refs(field):
    _, batch, _ = inputs()
    kwargs = dict(
        prediction_id='prediction-1', values=batch.values[..., 0],
        time_axis=tuple(batch.time_axis.values), asset_axis=tuple(batch.asset_axis.values),
        model_artifact_ref='model:M1', feature_manifest_ref='features:FS1',
        oos_evidence_ref='fold:OOS1',
    )
    kwargs[field] = '   '
    with pytest.raises(EvaluationContractError, match=f'{field} is required'):
        ModelPredictionArtifact(**kwargs)


@pytest.mark.parametrize('times', [(2, 1, 3), (1, 1, 2)])
def test_prediction_artifact_requires_strict_monotonic_time_axis(times):
    with pytest.raises(EvaluationContractError, match='ordered and unique|strictly increasing'):
        ModelPredictionArtifact(
            prediction_id='p', values=np.zeros((3, 2)), time_axis=times,
            asset_axis=('a', 'b'), model_artifact_ref='m',
            feature_manifest_ref='f', oos_evidence_ref='oos',
        )
