"""Runtime-bound inputs must never become serialized metric variants."""

RUNTIME_INPUT_PARAMETERS = frozenset({
    "factor_batch", "label_bundle", "computed_metrics", "metadata",
    "factor_values", "forward_returns", "returns", "validity_mask",
    "calendar_snapshot", "time_index", "factor_ids",
    "daily_quantile_artifact", "_bind_request_inputs",
})
