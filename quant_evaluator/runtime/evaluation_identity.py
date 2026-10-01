"""Pure builders and cache-residency probes for evaluation identity."""
from collections.abc import Mapping

from quant_evaluator.contracts import array_identity
from quant_evaluator.contracts import _hashutil


def build_evaluation_config_hash_fields(
    factor_batch, label_bundle, metric_ids, *, metric_versions: Mapping[str, str],
    metric_parameters, context, quantile_builder_parameters, portfolio_returns,
    holding_returns, portfolio_spec, trade_eligibility, calendar_snapshot,
    exposure_panel, generalization_evidence, split_ref, request_fields,
):
    """Build canonical EvaluationConfig.v2 fields from resolved inputs.

    Metric versions are supplied by the evaluator boundary so this module
    remains independent of registry lookup and evaluator imports.
    """
    return {
        "metrics": metric_ids, "versions": metric_versions,
        "parameters": metric_parameters,
        "label_hash": label_bundle.content_hash,
        "factor_ids": factor_batch.factor_ids,
        "factor_values": factor_batch.values,
        "factor_validity": factor_batch.validity,
        "context": context,
        "quantile_builder_parameters": quantile_builder_parameters,
        "calendar_snapshot": None if calendar_snapshot is None else {
            "id": calendar_snapshot.snapshot_id, "market": calendar_snapshot.market,
            "timezone": calendar_snapshot.timezone,
            "source": calendar_snapshot.source_version,
            "trading_days": calendar_snapshot.trading_days,
            "sessions": calendar_snapshot.sessions,
            "early_close": calendar_snapshot.early_close,
        },
        "portfolio_returns": (
            portfolio_returns.to_dict()
            if portfolio_returns is not None and holding_returns is None else None
        ),
        "holding_returns": holding_returns.content_hash if holding_returns is not None else None,
        "portfolio_spec": portfolio_spec.to_dict() if portfolio_spec is not None else None,
        "trade_eligibility": (
            trade_eligibility.content_hash if trade_eligibility is not None else None
        ),
        "exposure_panel": exposure_panel.to_dict() if exposure_panel is not None else None,
        "generalization_evidence": (
            generalization_evidence.to_dict()
            if generalization_evidence is not None else None
        ),
        "split_ref": split_ref.to_dict() if split_ref is not None else None,
        **request_fields,
    }


def evaluation_identity_cache_ready(*, raw_arrays, fields, tag, array_keys) -> bool:
    """Return true only when every raw and canonical-stream identity is resident.

    Probes are non-promoting and fail closed on misses, unknown arrays, or
    unsupported metadata. They never hash a cold payload to answer readiness.
    """
    try:
        if any(array_identity.authoritative_array_hash_cache_state(array) is not True
               for array in raw_arrays):
            return False
        return _hashutil.streamed_arrays_cache_ready(
            tag=tag, fields=fields, array_keys=array_keys,
        ) is True
    except Exception:
        return False
