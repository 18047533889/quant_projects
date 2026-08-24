# -*- coding: utf-8 -*-
"""Checkpoint store and segmented-incremental integration tests (audit §11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.production_hardening import SEGMENTED_EXECUTION_CANONICALS
from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.runtime.stateful_incremental import (
    segmented_incremental_available,
    stateful_canonicals_in_ir,
    try_stateful_segmented_incremental,
)
from factor_engine.stateful_contract import StatefulCheckpointRegistry
from factor_engine.stateful_runtime import execute_stateful_segment


class _PanelSource:
    """Minimal DataSource stub returning MultiIndex (timestamp, instrument) series."""

    def __init__(self, panel: pd.DataFrame):
        self._panel = panel

    def load_column(self, name: str) -> pd.Series:
        stacked = self._panel.stack()
        stacked.index = stacked.index.set_names(["timestamp", "instrument"])
        return stacked.rename(name)


def _ema_ir(span: int = 3) -> IRNode:
    return IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})


def _close_panel(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "A": np.arange(n, dtype=float) + 10.0,
            "B": np.arange(n, dtype=float) * 2.0 + 1.0,
        },
        index=idx,
    )


def test_segmented_available_only_for_single_segmented_root() -> None:
    assert segmented_incremental_available(ir=_ema_ir())
    # A non-stateful root is ineligible.
    mixed = IRNode(op="ts_delta", inputs=(_ema_ir(),), attrs={"n": 1})
    assert not segmented_incremental_available(ir=mixed)
    # A stateful root that is not segmented-execution-supported is ineligible.
    unsupported = IRNode(op="KAMA", inputs=(IRNode(op="column", attrs={"name": "close"}),))
    assert not segmented_incremental_available(ir=unsupported)
    assert "ts_ema" in stateful_canonicals_in_ir(mixed)


def test_bootstrap_and_resume_match_full_history_parity(tmp_path) -> None:
    n, split = 40, 25
    panel = _close_panel(n)

    full_ref: dict[str, np.ndarray] = {}
    for inst in panel.columns:
        res = execute_stateful_segment(
            "ts_ema", {"x": panel[inst].to_numpy(dtype=float)},
            timestamps=panel.index, instrument=inst,
            input_identity={"dataset": "unit"}, params={"span": 3},
            starts_at_dataset_origin=True,
        )
        full_ref[inst] = res.values

    store = StatefulCheckpointStore(root=tmp_path)

    # Bootstrap over [0, split] (inclusive): the checkpoint is persisted at the
    # state one bar before the segment end, i.e. at index[split-1], so the
    # following run (which re-computes the terminal bar, 1-bar overlap) can
    # resume from index[split] exactly.
    bootstrap = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(), source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert bootstrap is not None
    boot_series, mode = bootstrap
    assert mode["bootstrap"] is True
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        np.column_stack([full_ref["A"][: split + 1], full_ref["B"][: split + 1]]),
        equal_nan=True,
    )

    # Incremental resume over [split, n-1] from the persisted checkpoint.
    tail = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(),
        source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, mode2 = tail
    assert mode2["bootstrap"] is False
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        np.column_stack([full_ref["A"][split:], full_ref["B"][split:]]),
        equal_nan=True,
    )


def test_missing_checkpoint_falls_back_to_none(tmp_path) -> None:
    store = StatefulCheckpointStore(root=tmp_path)
    panel = _close_panel(10)
    # No checkpoint exists -> incremental attempt must return None (fall back).
    attempt = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(), source=_PanelSource(panel),
        store=store, start=panel.index[5], end=panel.index[-1], bootstrap=False,
    )
    assert attempt is None


def test_checkpoint_store_persists_and_gates_by_fingerprint(tmp_path) -> None:
    store = StatefulCheckpointStore(root=tmp_path)
    panel = _close_panel(10)
    res = execute_stateful_segment(
        "ts_ema", {"x": panel["A"].to_numpy(dtype=float)},
        timestamps=panel.index, instrument="A",
        input_identity={"factor_id": "f1", "params": {"span": 3}},
        params={"span": 3}, starts_at_dataset_origin=True,
    )
    store.save("f1", res.checkpoint)

    # The checkpoint as_of is panel.index[-1]; a cutoff strictly after it loads.
    later = panel.index[-1] + pd.Timedelta(days=1)
    loaded = store.load_latest("f1", "ts_ema", "A", before=later)
    assert loaded is not None
    assert loaded.as_of == res.checkpoint.as_of
    # A checkpoint at/after the cutoff is not usable for a segment starting there.
    assert store.load_latest("f1", "ts_ema", "A", before=panel.index[-1]) is None
    # Wrong operator / instrument / factor ids resolve to nothing.
    assert store.load_latest("f2", "ts_ema", "A", before=panel.index[9]) is None
    assert store.load_latest("f1", "MACD_line", "A", before=panel.index[9]) is None
    assert store.load_latest("f1", "ts_ema", "B", before=panel.index[9]) is None


def _make_source(n: int) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["timestamp", "instrument"])
    return pd.Series([float(i + 1) for i in range(len(idx))], index=idx)


def test_run_incremental_bootstraps_then_resumes_from_checkpoint(tmp_path) -> None:
    from factor_engine.api import ts_ema
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.materializer import ParquetMaterializer
    from tests.helpers import InMemorySeriesSource

    lake = tmp_path / "lake"
    factor = Factor(name="ema3", expr=ts_ema(col("close"), 3))

    def engine_for(n: int) -> FactorEngine:
        return FactorEngine(
            backend=PandasBackend(),
            data_source=InMemorySeriesSource(data={"close": _make_source(n)}),
        )

    # Phase A: full history materialized at the 20-bar watermark.
    eng_a = engine_for(20)
    mat = ParquetMaterializer(lake_root=lake)
    mat.materialize(factor_id="ema3", result=eng_a.run(factor)["result"], ast_hash="h1")

    # Phase B: 20 new bars -> bootstrap seeds the checkpoint, parity holds.
    eng_b = engine_for(40)
    full_b = eng_b.run(factor)["result"]
    out_b = eng_b.run_incremental(factor, factor_id="ema3", lake_root=lake, lookback_extra=0)
    assert out_b["incremental"]["mode"] == "stateful_segmented"
    assert out_b["incremental"]["bootstrap"] is True
    overlap_b = out_b["result"].index.intersection(full_b.index)
    pd.testing.assert_series_equal(
        out_b["result"].loc[overlap_b], full_b.loc[overlap_b], check_names=False
    )
    mat.materialize(factor_id="ema3", result=out_b["result"], ast_hash="h1")

    # Phase C: 20 more bars -> resume from the persisted checkpoint (no bootstrap).
    eng_c = engine_for(60)
    full_c = eng_c.run(factor)["result"]
    out_c = eng_c.run_incremental(factor, factor_id="ema3", lake_root=lake, lookback_extra=0)
    assert out_c["incremental"]["mode"] == "stateful_segmented"
    assert out_c["incremental"]["bootstrap"] is False
    assert len(out_c["result"]) > 0
    overlap_c = out_c["result"].index.intersection(full_c.index)
    pd.testing.assert_series_equal(
        out_c["result"].loc[overlap_c], full_c.loc[overlap_c], check_names=False
    )


def test_run_incremental_stateful_falls_back_for_unsupported_root(tmp_path) -> None:
    # A factor whose root is not a segmented-execution canonical is not
    # checkpoint-eligible and must run the standard path, still matching full.
    from factor_engine.api import rank, ts_mean
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.materializer import ParquetMaterializer
    from tests.helpers import InMemorySeriesSource

    lake = tmp_path / "lake2"
    factor = Factor(name="mom2", expr=rank(ts_mean(col("close"), 2)))

    def engine_for(n: int) -> FactorEngine:
        return FactorEngine(
            backend=PandasBackend(),
            data_source=InMemorySeriesSource(data={"close": _make_source(n)}),
        )

    eng_a = engine_for(20)
    mat = ParquetMaterializer(lake_root=lake)
    mat.materialize(factor_id="mom2", result=eng_a.run(factor)["result"], ast_hash="h1")
    eng_b = engine_for(40)
    full_b = eng_b.run(factor)["result"]
    out_b = eng_b.run_incremental(factor, factor_id="mom2", lake_root=lake, lookback_extra=0)
    assert out_b["incremental"].get("mode") is None  # standard path
    overlap = out_b["result"].index.intersection(full_b.index)
    pd.testing.assert_series_equal(
        out_b["result"].loc[overlap], full_b.loc[overlap], check_names=False
    )


def test_segmented_canonicals_all_have_runtime_branches() -> None:
    # Every segmented-execution canonical must be registered in the stateful
    # checkpoint registry AND resolvable through the segmented entry point.
    for canonical in SEGMENTED_EXECUTION_CANONICALS:
        assert StatefulCheckpointRegistry.get(canonical) is not None, canonical
        assert StatefulCheckpointRegistry.get(canonical).segmented_execution_supported, canonical
