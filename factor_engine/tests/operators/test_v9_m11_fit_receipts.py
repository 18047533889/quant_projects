from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np

from factor_engine.cleaned_operators.ts_model import _rolling_core as rc


def _failure(reason):
    def fit(_design, _y):
        rc._set_fit_status(False, reason)
        return None
    return fit


def _owned_scope(instrument="AAA"):
    return rc.FitScope(
        canonical="ts_test", backend="pandas_numpy", profile="research",
        instrument=instrument, execution_id="exec-1", run_id="run-1",
        task_id="task-1", factor_id="factor-1", maturity_cutoff=0)


def test_fit_result_captures_success_then_failure_without_stale_status():
    design = np.column_stack((np.ones(6), np.arange(6.0)))
    success = rc.fit_result(rc.ols_fit, design, np.arange(6.0))
    failure = rc.fit_result(rc.ols_fit, np.ones((3, 2)), np.arange(3.0))

    assert success.value is not None
    assert success.status.converged is True
    assert success.status.reason == "converged"
    assert failure.value is None
    assert failure.status.converged is False
    assert failure.status.reason == "singular"

    missing_status = rc.fit_result(lambda _x, _y: np.array([1.0]), design, np.arange(6.0))
    assert missing_status.status.reason == "unknown_status"
    sink = rc.BoundedFitFailureSink()
    sink.record(missing_status, _owned_scope())
    assert sink.unknown_status_count == 1
    assert sink.page() == ()

    def contradictory(_x, _y):
        rc._set_fit_status(True, "converged")
        return None

    mismatch = rc.fit_result(contradictory, design, np.arange(6.0))
    sink.record(mismatch, _owned_scope())
    assert mismatch.status.reason == "internal_status_mismatch"
    assert sink.unknown_status_count == 2


def test_rolling_fit_consumes_failure_with_explicit_window_scope():
    sink = rc.BoundedFitFailureSink(detail_capacity=10)
    y = np.arange(8.0)
    x = np.arange(8.0) ** 2
    rc.rolling_fit(
        y, [x], 4, 3, fit_fn=_failure("forced"),
        fit_scope=_owned_scope(), failure_sink=sink)

    receipts = sink.page()
    assert receipts
    receipt = next(r for r in receipts if r.status.reason == "forced")
    assert receipt.scope_kind == "factor_window"
    assert receipt.scope.instrument == "AAA"
    assert receipt.scope.canonical == "ts_test"
    assert receipt.scope.window_start == 0
    assert receipt.scope.window_end == 2
    assert receipt.scope.output_row == 2
    assert receipt.scope.fit_cutoff == 2
    assert receipt.scope.maturity_cutoff == 0


def test_missing_authority_is_kernel_only_and_never_inferred():
    sink = rc.BoundedFitFailureSink()
    sink.record(rc.FitResult(None, rc.FitStatus(False, "forced")),
                rc.FitScope(canonical="ts_test", instrument="AAA"))
    receipt = sink.page()[0]
    assert receipt.scope_kind == "kernel_only"
    assert receipt.scope.factor_id is None
    assert receipt.scope.run_id is None

    invalid_window = rc.FitScope(
        canonical="ts_test", backend="pandas_numpy", profile="research",
        instrument="AAA", window_start=5, window_end=4, output_row=6,
        fit_cutoff=4, maturity_cutoff=3, execution_id="exec-1",
        run_id="run-1", task_id="task-1", factor_id="factor-1")
    assert invalid_window.scope_kind == "kernel_only"

    for bad in (
        rc.FitScope(**{**invalid_window.__dict__, "window_start": [1]}),
        rc.FitScope(**{**invalid_window.__dict__, "window_start": 1,
                       "window_end": 2, "fit_cutoff": 7}),
        rc.FitScope(**{**invalid_window.__dict__, "window_start": 1,
                       "window_end": 2, "maturity_cutoff": 7}),
    ):
        assert bad.scope_kind == "kernel_only"


def test_detail_capacity_and_group_capacity_are_bounded():
    sink = rc.BoundedFitFailureSink(detail_capacity=2, group_capacity=1)
    failed = rc.FitResult(None, rc.FitStatus(False, "forced"))
    for i in range(5):
        sink.record(failed, _owned_scope(str(i)))
    sink.record(rc.FitResult(None, rc.FitStatus(False, "other")), _owned_scope())

    assert [r.sequence for r in sink.page()] == [5, 6]
    assert sink.dropped_details == 4
    assert sink.counts() == {("ts_test", "forced"): 5}
    assert sink.overflow_group_count == 1
    assert [r.sequence for r in sink.page(after_sequence=5, limit=1)] == [6]


def test_capacity_parameters_reject_bool_and_float():
    for detail, groups in ((True, 1), (1.0, 1), (1, False), (1, 2.0)):
        try:
            rc.BoundedFitFailureSink(detail_capacity=detail, group_capacity=groups)
        except ValueError:
            pass
        else:
            raise AssertionError("non-integer capacity was accepted")


def test_status_details_are_detached_from_mutable_inputs():
    mutable = {"items": [1, 2]}
    rc._set_fit_status(False, "forced", payload=mutable)
    snapshot = rc.last_fit_status()
    direct = rc.FitStatus(False, "forced", (("payload", mutable),))
    mutable["items"].append(3)
    expected = (("items", (1, 2)),)
    assert snapshot["payload"] == expected
    assert direct.as_dict()["payload"] == expected


def test_status_detail_cycles_and_large_values_are_bounded():
    cyclic = []
    cyclic.append(cyclic)
    rc._set_fit_status(False, "forced", cycle=cyclic)
    assert rc.last_fit_status()["cycle"] == ("<cycle>",)

    rc._set_fit_status(False, "forced", array=np.arange(100_000))
    array = rc.last_fit_status()["array"]
    assert array[0:3] == ("<ndarray>", (100_000,), "int64")
    assert len(array[3]) <= rc._DETAIL_MAX_ITEMS + 1
    assert array[3][-1] == "<truncated>"

    rc._set_fit_status(False, "forced", text="sensitive" * 1_000)
    text = rc.last_fit_status()["text"]
    assert text[1] == "<truncated>"
    assert len(text[0]) == rc._DETAIL_MAX_STRING

    class Secret:
        def __repr__(self):
            raise AssertionError("arbitrary repr must not be called")

    rc._set_fit_status(False, "forced", opaque=Secret())
    assert rc.last_fit_status()["opaque"][0] == "<object_type>"


def test_reason_and_retained_scope_are_bounded_and_sanitized():
    long_reason = "failure-" * 1_000
    status = rc.FitStatus(False, long_reason)
    assert status.reason.endswith("<truncated>")
    assert len(status.reason) <= rc._DETAIL_MAX_STRING + len("<truncated>")

    class Reason:
        def __str__(self):
            raise AssertionError("arbitrary str must not be called")

    rc._set_fit_status(False, Reason())
    assert rc.last_fit_status()["reason"].startswith("<reason_type:")

    huge_array = np.arange(100_000)
    unsafe = rc.FitScope(canonical=["not", "hashable"], window_start=huge_array)
    sink = rc.BoundedFitFailureSink()
    sink.record(rc.FitResult(None, status), unsafe)
    receipt = sink.page()[0]
    assert receipt.scope_kind == "kernel_only"
    assert receipt.scope.canonical is None
    assert receipt.scope.window_start is None
    assert sink.counts() == {("<kernel_only>", status.reason): 1}


def test_pre_solver_failures_are_receipted_without_changing_outputs():
    sink = rc.BoundedFitFailureSink(detail_capacity=20)
    beta, resid, resid_std = rc.rolling_fit(
        np.array([1.0, np.nan, 3.0, 4.0]), [np.arange(4.0)], 3, 3,
        fit_scope=_owned_scope(), failure_sink=sink)
    assert np.isnan(beta).all() and np.isnan(resid).all() and np.isnan(resid_std).all()
    assert sink.counts()[("ts_test", "insufficient_sample")] >= 1

    singular = rc.BoundedFitFailureSink()
    rc.rolling_fit(np.arange(5.0), [np.ones(5)], 3, 3,
                   fit_scope=_owned_scope(), failure_sink=singular)
    assert singular.counts()[("ts_test", "singular")] >= 1


def test_context_bound_sinks_are_isolated_across_threads():
    def run(reason):
        sink = rc.BoundedFitFailureSink()
        with rc.fit_failure_receipts(sink):
            rc.rolling_fit(np.arange(5.0), [np.arange(5.0) ** 2], 3, 3,
                           fit_fn=_failure(reason), fit_scope=_owned_scope(reason))
        return {r.status.reason for r in sink.page() if r.status.reason != "insufficient_sample"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("thread-a", "thread-b")))
    assert results == [{"thread-a"}, {"thread-b"}]


def test_one_inherited_sink_is_safe_when_shared_by_threads():
    sink = rc.BoundedFitFailureSink(detail_capacity=8)
    failed = rc.FitResult(None, rc.FitStatus(False, "shared"))

    def record_many(_):
        for _ in range(100):
            sink.record(failed, _owned_scope())

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record_many, range(4)))
    assert sink.counts() == {("ts_test", "shared"): 400}
    assert len(sink.page()) == 8
    assert sink.dropped_details == 392


def test_context_bound_sinks_are_isolated_across_async_tasks():
    async def run(reason):
        sink = rc.BoundedFitFailureSink()
        with rc.fit_failure_receipts(sink):
            await asyncio.sleep(0)
            rc.rolling_fit(np.arange(5.0), [np.arange(5.0) ** 2], 3, 3,
                           fit_fn=_failure(reason), fit_scope=_owned_scope(reason))
        return {r.status.reason for r in sink.page() if r.status.reason != "insufficient_sample"}

    async def gather():
        return await asyncio.gather(run("async-a"), run("async-b"))

    assert asyncio.run(gather()) == [{"async-a"}, {"async-b"}]


def test_explicit_sink_wins_and_nested_context_restores_after_exception():
    outer = rc.BoundedFitFailureSink()
    inner = rc.BoundedFitFailureSink()
    explicit = rc.BoundedFitFailureSink()
    y = np.arange(4.0)
    x = y ** 2
    with rc.fit_failure_receipts(outer):
        try:
            with rc.fit_failure_receipts(inner):
                rc.rolling_fit(y, [x], 3, 3, fit_fn=_failure("explicit"),
                               fit_scope=_owned_scope(), failure_sink=explicit)
                raise RuntimeError("restore")
        except RuntimeError:
            pass
        rc.rolling_fit(y, [x], 3, 3, fit_fn=_failure("outer"),
                       fit_scope=_owned_scope())

    assert inner.page() == ()
    assert "explicit" in {r.status.reason for r in explicit.page()}
    assert "outer" in {r.status.reason for r in outer.page()}


def test_owner_scope_transport_merges_coordinates_and_restores_nested_scope():
    outer = _owned_scope()
    inner = rc.FitScope(canonical="inner")
    sink = rc.BoundedFitFailureSink()
    failed = rc.FitResult(None, rc.FitStatus(False, "forced"))

    assert rc.current_fit_scope() is None
    with rc.fit_receipt_scope(outer):
        assert rc.current_fit_scope().canonical == "ts_test"
        try:
            with rc.fit_receipt_scope(inner):
                assert rc.current_fit_scope().canonical == "inner"
                raise RuntimeError("restore")
        except RuntimeError:
            pass
        assert rc.current_fit_scope().canonical == "ts_test"
        rc.record_current_fit(
            failed, instrument="AAA", window_start=1, window_end=3,
            output_row=4, fit_cutoff=3, maturity_cutoff=2,
            failure_sink=sink)
    assert rc.current_fit_scope() is None
    receipt = sink.page()[0]
    assert receipt.scope_kind == "factor_window"
    assert receipt.scope.canonical == "ts_test"
    assert receipt.scope.window_start == 1
