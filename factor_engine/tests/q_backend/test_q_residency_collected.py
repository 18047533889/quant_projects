"""Collected coverage for automatic q region residency."""

from backend.q_backend.q_compiler import QRegionPlan
from backend.q_backend.q_executor import (
    get_q_executor,
    get_q_executor_telemetry,
    reset_q_executor_telemetry,
)
from backend.q_backend.test_q_residency import (
    mock_q_process,
    mock_type_adapter,
    reset_q_executor_singleton,
    sample_input_data,
)


def test_batch_regions_automatic_residency(
    mock_q_process,
    mock_type_adapter,
    sample_input_data,
    reset_q_executor_singleton,
):
    """Consecutive q regions reuse resident intermediates until final output."""
    reset_q_executor_telemetry()
    executor = get_q_executor()
    plans = [
        QRegionPlan(
            region_id="region_A",
            node_ids=("node_1",),
            q_code="temp1: mavg[20; input_table[`price]]",
            input_tables=("input_table",),
            output_table="temp1",
        ),
        QRegionPlan(
            region_id="region_B",
            node_ids=("node_2",),
            q_code="temp2: mavg[10; temp1]",
            input_tables=("temp1",),
            output_table="temp2",
        ),
        QRegionPlan(
            region_id="region_C",
            node_ids=("node_3",),
            q_code="final: mavg[5; temp2]",
            input_tables=("temp2",),
            output_table="final",
        ),
    ]

    results = executor.execute_batch_regions(
        plans,
        {"input_table": sample_input_data},
        enable_residency=True,
    )

    assert len(results) == 3
    assert all(result.success for result in results)
    assert results[0].output_df.empty
    assert results[1].output_df.empty
    assert not results[2].output_df.empty
    assert mock_type_adapter.q_to_pandas.call_count == 1

    telemetry = get_q_executor_telemetry()
    assert telemetry["resident_reuse_count"] == 2
    assert telemetry["python_to_q_bytes"] > 0
    assert telemetry["q_to_python_bytes"] > 0
