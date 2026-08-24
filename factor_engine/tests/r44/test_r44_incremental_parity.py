# -*- coding: utf-8 -*-
"""R44 node-level incremental FactorEngine — parity + destructive-scenario tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.incremental_parity import (
    DEFAULT_PARAMS,
    IncrementalParityChecker,
    SEGMENTED_CANONICALS,
    build_incremental_e2e_certificate,
    default_panel_factory,
    run_destructive_scenarios,
)
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore


def _store(tmp_path) -> StatefulCheckpointStore:
    return StatefulCheckpointStore(root=tmp_path)


def _run_parity(canonical: str, params: dict, chunks, tmp_path, n: int = 40, split: int = 20):
    checker = IncrementalParityChecker()
    panel = default_panel_factory(n)
    store = _store(tmp_path)
    return checker.assert_parity(
        canonical, params, panel, chunks, store, split=split
    )


def test_ts_ema_full_vs_one_day_vs_chunks_parity(tmp_path) -> None:
    result = _run_parity(
        "ts_ema", {"span": 3}, [1], tmp_path, n=40, split=20
    )
    assert result.full_vs_incremental_equal
    assert result.restart_resume_passed
    assert result.max_abs_diff <= 1e-9
    assert any(ck == 1 and ok for ck, ok in result.chunks_tested)

    result_chunked = _run_parity(
        "ts_ema", {"span": 3}, [17, 3, 91], tmp_path, n=40, split=20
    )
    assert result_chunked.full_vs_incremental_equal
    assert result_chunked.restart_resume_passed
    sizes = {ck for ck, _ok in result_chunked.chunks_tested}
    assert sizes <= {0, 17, 3, 91}


def test_restart_resume_equals_reference(tmp_path) -> None:
    checker = IncrementalParityChecker()
    panel = default_panel_factory(40)
    store = _store(tmp_path)
    ref = checker.full_reference("ts_ema", {"span": 3}, panel)
    result = checker.assert_parity(
        "ts_ema", {"span": 3}, panel, [1], store, split=20, restart_crash_day=26
    )
    assert result.restart_resume_passed
    assert result.full_vs_incremental_equal
    # ref 非空
    assert ref.shape[0] == 40


def test_at_least_one_ewm_or_macd_canonical_parity_or_not_run(tmp_path) -> None:
    checker = IncrementalParityChecker()
    panel = default_panel_factory(40)
    checked_any = False
    for canonical in ("ts_ewm_std", "ts_ewm_var", "MACD_line"):
        if canonical not in SEGMENTED_CANONICALS:
            continue
        params = DEFAULT_PARAMS[canonical]
        try:
            result = checker.assert_parity(
                canonical, params, panel, [7], _store(tmp_path), split=20
            )
        except Exception as exc:
            # unsupported / 无法 bootstrap -> 显式 NOT_RUN，绝不伪造 PASS
            msg = str(exc).lower()
            if (
                "unsupported" in msg
                or "not implemented" in msg
                or "bootstrap failed" in msg
            ):
                continue
            raise
        checked_any = True
        assert result.full_vs_incremental_equal, f"{canonical} parity failed"
        assert result.restart_resume_passed, f"{canonical} restart failed"
    if not checked_any:
        pytest.skip("no EWM/MACD canonical supported -> NOT_RUN")


def test_destructive_runner_completes_and_certificate_has_all_keys(tmp_path) -> None:
    scenarios = run_destructive_scenarios(
        store_factory=lambda: _store(tmp_path),
        canonical="ts_ema", params={"span": 3},
    )
    assert len(scenarios) >= 8, f"only {len(scenarios)} scenarios ran"
    for sc in scenarios:
        assert {"name", "ok", "reason", "skipped"} <= set(sc), sc
        if sc.get("skipped"):
            assert sc.get("reason") == "NOT_RUN" or sc.get("reason"), sc
    # 至少 8 个场景运行（ok 或 skipped）
    ran = [s for s in scenarios if not s.get("skipped") or s.get("reason") != "NOT_RUN"]
    assert len(ran) >= 8, f"expected >=8 scenarios to run, got {len(ran)}"

    parity_results = [_run_parity("ts_ema", {"span": 3}, [1], tmp_path)]
    report = build_incremental_e2e_certificate(parity_results, scenarios)
    for key in (
        "operator_total", "TRUE_INCREMENTAL", "TAIL_REPLAY", "EVENT_INCREMENTAL",
        "FULL_REPLAY_ONLY", "NOT_CERTIFIED", "checkpoint_resumable_node_count",
        "cross_factor_shared_state_ratio", "per_day_rows_read",
        "per_day_incremental_compute_time", "full_vs_incremental_speedup",
        "full_vs_incremental_numerical_parity", "revision_replay_correctness",
        "state_bytes", "factor_bytes", "TTDC",
    ):
        assert key in report, f"missing machine-report key: {key}"


def test_corrupted_and_stale_checkpoint_fail_closed(tmp_path) -> None:
    from pathlib import Path
    from factor_engine.runtime.incremental_parity import _PanelSource, _segmented_ir
    from factor_engine.runtime.stateful_incremental import try_stateful_segmented_incremental
    from stateful_contract import StateCheckpoint

    panel = default_panel_factory(40)
    split = 20
    src = _PanelSource(panel)
    ir = _segmented_ir("ts_ema", {"span": 3})

    # ---- corrupted checkpoint -> fail-closed None ----
    store = _store(tmp_path)
    try_stateful_segmented_incremental(
        factor_id="c1", ir=ir, source=src, store=store,
        start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    factor_dir = Path(str(store.root)) / "c1"
    assert list(factor_dir.glob("*.json")), "expected a checkpoint file"
    for path in factor_dir.glob("*.json"):
        path.write_text("{not valid", encoding="utf-8")
    seg = try_stateful_segmented_incremental(
        factor_id="c1", ir=ir, source=src, store=store,
        start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert seg is None, "corrupted checkpoint must fail closed to None"

    # ---- stale checkpoint (as_of one day behind) -> fail-closed None ----
    store2 = _store(tmp_path)
    try_stateful_segmented_incremental(
        factor_id="c2", ir=ir, source=src, store=store2,
        start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    fdir2 = Path(str(store2.root)) / "c2"
    for path in fdir2.glob("*.json"):
        cp = StateCheckpoint.from_json(path.read_text(encoding="utf-8"))
        cp = StateCheckpoint(
            operator=cp.operator, instrument=cp.instrument,
            as_of=str(pd.Timestamp(cp.as_of) - pd.Timedelta(days=1)),
            state_schema_version=cp.state_schema_version,
            semantic_version=cp.semantic_version,
            input_fingerprint=cp.input_fingerprint, state=cp.state,
        )
        path.write_text(cp.to_json(), encoding="utf-8")
    seg2 = try_stateful_segmented_incremental(
        factor_id="c2", ir=ir, source=src, store=store2,
        start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert seg2 is None, "stale checkpoint must fail closed to None"
