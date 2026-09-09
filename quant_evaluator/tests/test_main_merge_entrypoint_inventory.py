from quant_evaluator.metrics.coverage_compiler import canonical_entrypoint_inventory


def test_inventory_binds_actual_registry_parameters_and_packaging():
    rows = canonical_entrypoint_inventory()
    assert len(rows) > 50
    bound = {row["metric_id"]: row for row in rows if row.get("implementation")}
    for name in ("rank_ic", "sharpe_ratio", "max_drawdown"):
        row = bound[name]
        assert row["packaged"]
        assert row["parameter_contract"]
        assert len(row["source_sha256"]) == 64
        assert row["static_test_references"]
        assert row["golden_execution_status"] == "NOT_RUN"
