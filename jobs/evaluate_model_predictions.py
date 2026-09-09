"""Job-level QE/modeling composition; both domain and DTO packages stay independent."""

import numpy as np


def evaluate_prediction_evidence(batch, labels, *, min_assets, portfolio_returns=None):
    from quant_evaluator.runtime.evaluator import evaluate
    from quant_evaluator.metrics.ic_summary import compute_icir

    predictive = evaluate(batch, labels, metrics=['rank_ic', 'rank_ic_series'],
        metric_parameters={name: {'min_assets': min_assets}
                           for name in ('rank_ic', 'rank_ic_series')})
    ir = float(compute_icir(predictive.artifacts['rank_ic_series'].values, min_periods=2)[0])
    portfolio = None
    if portfolio_returns is not None:
        portfolio = evaluate(batch, labels, metrics=['sharpe_ratio', 'max_drawdown'],
                             portfolio_returns=portfolio_returns)
    return predictive, portfolio, ir


def evaluate_model_predictions(*args, **kwargs):
    """Existing model report with the production composition's QE evidence port."""
    from modeling.evaluation import evaluate_predictions
    prediction_artifact = kwargs.get('prediction_artifact')
    if prediction_artifact is not None:
        from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
        labels = kwargs.get('label_bundle')
        if labels is None:
            raise ValueError('prediction_artifact requires label_bundle')
        if tuple(labels.decision_time) != prediction_artifact.time_axis:
            raise ValueError('label decision axis does not match prediction artifact')
        if labels.asset_axis is None or tuple(labels.asset_axis.values) != prediction_artifact.asset_axis:
            raise ValueError('label asset axis does not match prediction artifact')
        # Preserve the label contract's canonical coordinate representation at
        # the QE boundary after proving it is identical to the prediction axes.
        time_values = np.asarray(labels.decision_time)
        asset_values = np.asarray(labels.asset_axis.values)
        time_axis = AxisRef('time', str(time_values.dtype), len(time_values), time_values)
        asset_axis = AxisRef('asset', str(asset_values.dtype), len(asset_values), asset_values)
        batch = FactorBatch((prediction_artifact.prediction_id,), time_axis, asset_axis,
                            prediction_artifact.values[..., None])
        kwargs['prediction_id'] = prediction_artifact.prediction_id
        kwargs['prediction_batch'] = batch
        if not args:
            args = (prediction_artifact.values.ravel(), labels.values.ravel(),
                    np.repeat(prediction_artifact.time_axis, len(prediction_artifact.asset_axis)),
                    np.tile(prediction_artifact.asset_axis, len(prediction_artifact.time_axis)))
    if 'prediction_evaluator' in kwargs:
        raise ValueError('platform owns the prediction evaluation port')
    return evaluate_predictions(*args, **kwargs, prediction_evaluator=evaluate_prediction_evidence)


def train_evaluate_model_predictions(
    learner_cls, train_ds, validation_ds, oos_ds, label_bundle, *,
    train_kwargs, training_importance_evidence=None, oos_ablation_evidence_refs=None,
    portfolio_returns=None,
):
    """Real training -> ordered OOS prediction artifact -> QE report composition."""
    from modeling.trainer import train_model

    result = train_model(learner_cls, train_ds, validation_ds, **dict(train_kwargs))
    artifact, report = evaluate_trained_model_predictions(
        result, oos_ds, label_bundle,
        training_importance_evidence=training_importance_evidence,
        oos_ablation_evidence_refs=oos_ablation_evidence_refs,
        portfolio_returns=portfolio_returns,
    )
    return result, artifact, report


def evaluate_trained_model_predictions(
    result, oos_ds, label_bundle, *, training_importance_evidence=None,
    oos_ablation_evidence_refs=None, portfolio_returns=None,
):
    """Score one trained artifact on identity-bound OOS inputs and evaluate it."""
    from modeling.trainer import _dataset_schema_hash
    from modeling.artifact import PredictionContext
    from modeling.contracts import ApplicationWindow
    from modeling.evaluation import ModelPredictionArtifact
    from modeling.ledger import stable_hash

    if result.artifact is None:
        raise ValueError('training did not produce a model artifact')
    X, _, dates, assets, _ = oos_ds.as_matrix()
    ordered_times = tuple(dict.fromkeys(dates))
    ordered_assets = tuple(dict.fromkeys(assets))
    if (not ordered_times or not ordered_assets or
            len(dates) != len(ordered_times) * len(ordered_assets) or
            not np.array_equal(dates, np.repeat(ordered_times, len(ordered_assets))) or
            not np.array_equal(assets, np.tile(ordered_assets, len(ordered_times)))):
        raise ValueError('OOS panel must be a complete deterministic time-major asset grid')
    manifest = result.artifact.manifest
    if not manifest.artifact_id or not manifest.feature_schema_hash:
        raise ValueError('trained model lacks artifact or feature-manifest identity')
    oos_feature_schema_hash = _dataset_schema_hash(oos_ds)
    if oos_feature_schema_hash != manifest.feature_schema_hash:
        raise ValueError('OOS feature manifest does not match trained model')
    context = PredictionContext(
        application_window=ApplicationWindow(start=ordered_times[0], end=ordered_times[-1]),
        dates=np.asarray(dates), asof=ordered_times[-1],
        feature_schema_hash=oos_feature_schema_hash,
    )
    if not manifest.final_fit_end or context.dates[0] <= manifest.final_fit_end:
        raise ValueError('final model fit window must precede every OOS prediction time')
    predictions = result.artifact.predict(X, context=context).reshape(
        len(ordered_times), len(ordered_assets))
    if (tuple(label_bundle.decision_time) != ordered_times or
            label_bundle.asset_axis is None or
            tuple(label_bundle.asset_axis.values) != ordered_assets):
        raise ValueError('label axes do not match the ordered OOS panel')
    artifact_times = tuple(label_bundle.decision_time)
    artifact_assets = tuple(label_bundle.asset_axis.values)
    prediction_id = 'prediction:' + stable_hash({
        'model_artifact_ref': manifest.artifact_id,
        'feature_manifest_ref': oos_feature_schema_hash,
        'oos_evidence_ref': label_bundle.content_hash,
        'prediction_context': context.to_dict(),
        'asset_axis': list(artifact_assets),
        'oos_feature_values': np.asarray(X, dtype=np.float64).tolist(),
        'prediction_values': np.asarray(predictions, dtype=np.float64).tolist(),
    })
    artifact = ModelPredictionArtifact(
        prediction_id=prediction_id, values=predictions,
        time_axis=artifact_times, asset_axis=artifact_assets,
        model_artifact_ref=manifest.artifact_id,
        feature_manifest_ref=manifest.feature_schema_hash,
        oos_evidence_ref=label_bundle.content_hash,
    )
    report = evaluate_model_predictions(
        prediction_artifact=artifact, label_bundle=label_bundle,
        portfolio_returns=portfolio_returns,
        training_importance_evidence=(training_importance_evidence or
                                      result.training_importance_evidence),
        oos_ablation_evidence_refs=(oos_ablation_evidence_refs or
                                    result.oos_ablation_evidence_refs),
    )
    return artifact, report


__all__ = [
    'evaluate_prediction_evidence', 'evaluate_model_predictions',
    'train_evaluate_model_predictions', 'evaluate_trained_model_predictions',
]
