# -*- coding: utf-8 -*-
"""R40 bridge items #156-162/#203."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestCertificationHelpers:
    def _import_bridge(self):
        from backend import cleaned_bridge as cb

        return cb

    def test_bound_scalar_parameters_complete(self):
        cb = self._import_bridge()
        from cleaned_operators.common.data_cleaning import WindowMean

        op = WindowMean()
        idx = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["a", "b"])
        bound = cb._bound_scalar_parameters(op, [df], {})
        assert bound.complete is True
        # default window=20 resolved via the kernel defaults
        assert bound.normalized.get("window") == 20

    def test_input_dtype_signature_ordered(self):
        cb = self._import_bridge()
        idx = pd.date_range("2024-01-01", periods=5)
        a = pd.DataFrame(np.ones((5, 2), dtype="float64"), index=idx, columns=["a", "b"])
        b = pd.DataFrame(np.ones((5, 2), dtype="int64"), index=idx, columns=["c", "d"])
        sig = cb._input_dtype([a, b])
        assert isinstance(sig, cb.InputDTypeSignature)
        assert len(sig.inputs) == 4
        dtypes = [d.dtype for d in sig.inputs]
        assert "float64" in dtypes and "int64" in dtypes
        # key is a stable string
        assert isinstance(sig.to_key(), str)
        assert sig.to_key() != "no_inputs"

    def test_execution_variant_reflects_actual_backend(self, monkeypatch):
        cb = self._import_bridge()
        from cleaned_operators.common.data_cleaning import WindowMean

        op = WindowMean()
        variant = cb._execution_variant(op, "pandas_numpy", "window_mean")
        assert variant.backend == "pandas_numpy"
        assert variant.implementation_id != ""
        assert variant.code_hash != ""
        assert "window_mean" in variant.implementation_id or "WindowMean" in variant.implementation_id

    def test_operator_semantic_version_missing_fails_production(self, monkeypatch):
        cb = self._import_bridge()

        def _boom(canonical):
            raise RuntimeError("resolver down")

        monkeypatch.setattr("backend.operator_semantic_version.versioned_name", _boom)
        with pytest.raises(cb.ParameterCertificationInfrastructureError):
            cb._operator_semantic_version("ts_mean")

    def test_exception_types_exist(self):
        cb = self._import_bridge()
        assert issubclass(cb.NoCertifiedParameterRegionError, ValueError)
        assert issubclass(cb.ParameterCertificationInfrastructureError, RuntimeError)


class TestGrainTransformCertificate:
    def test_grain_transform_certificate_minute_to_daily(self):
        from backend import cleaned_bridge as cb

        cert = cb.GrainTransformCertificate(
            input_grain="minute", output_grain="daily",
            calendar_id="ashare", calendar_version="v2",
            timezone="Asia/Shanghai", session_id="s1",
            mapping_policy="session_end",
        )
        assert cert.output_grain == "daily"
        assert cert.calendar_version == "v2"
        assert cert.to_dict()["mapping_policy"] == "session_end"

    def test_grain_transform_rejects_duplicate_dates(self):
        from backend import cleaned_bridge as cb

        idx_in = pd.date_range("2024-01-01", periods=4)
        template = pd.DataFrame({"a": [0.0] * 4}, index=idx_in)
        idx_out = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"])
        out = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx_out)
        cert = cb.GrainTransformCertificate(
            input_grain="minute", output_grain="daily",
            calendar_id="", calendar_version="", timezone="", session_id="",
            mapping_policy="session_end",
        )
        errors = cb.validate_grain_transform(out, template, cert)
        assert any("not unique" in e for e in errors)

    def test_grain_transform_rejects_extra_columns(self):
        from backend import cleaned_bridge as cb

        idx_in = pd.date_range("2024-01-01", periods=4)
        template = pd.DataFrame({"a": [0.0] * 4}, index=idx_in)
        out = pd.DataFrame({"a": [1.0] * 4, "b": [2.0] * 4}, index=idx_in)
        cert = cb.GrainTransformCertificate(
            input_grain="minute", output_grain="daily",
            calendar_id="", calendar_version="", timezone="", session_id="",
            mapping_policy="session_end",
        )
        errors = cb.validate_grain_transform(out, template, cert)
        assert any("instrument axis" in e for e in errors)


class TestBackendRouteTelemetry:
    def test_concurrent_counts_not_lost(self):
        from backend import cleaned_bridge as cb
        from backend.panel_polars import ExecutionPerfCounters, set_request_perf_counters, reset_request_perf_counters

        counters = ExecutionPerfCounters()
        token = set_request_perf_counters(counters)
        try:
            import threading

            def worker(n):
                # Threads start with a fresh context in CPython, so each worker
                # binds the SAME counters object explicitly — the atomicity of
                # ExecutionPerfCounters.incr is what guarantees no lost counts.
                set_request_perf_counters(counters)

                class Ctx:
                    run_mode = "production"
                    runtime_stats = {}

                ctx = Ctx()
                for _ in range(100):
                    cb._record_backend_route(
                        ctx, canonical="ts_mean", backend="pandas_numpy", row_count_estimate=10
                    )

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert counters.get("backend_route:ts_mean:pandas_numpy") == 800
            assert counters.get("backend_route_total") == 800
        finally:
            reset_request_perf_counters(token)


class TestRepConversionCountersRequestScoped:
    def test_request_scoped_counters(self):
        from backend import panel_polars as pp
        from backend.panel_polars import ExecutionPerfCounters, set_request_perf_counters, reset_request_perf_counters

        idx = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame(np.eye(5), index=idx, columns=["a", "b", "c", "d", "e"])

        c1 = ExecutionPerfCounters()
        t1 = set_request_perf_counters(c1)
        try:
            plf = pp.panel_to_polars(df, use_arrow=True)
            out = pp.polars_to_panel(plf, template=df)
            assert pp.rep_conversion_bulk_count >= 1
        finally:
            reset_request_perf_counters(t1)

        # A fresh request starts at zero.
        c2 = ExecutionPerfCounters()
        t2 = set_request_perf_counters(c2)
        try:
            assert pp.rep_conversion_bulk_count == 0
            assert pp.rep_conversion_column_loop_count == 0
        finally:
            reset_request_perf_counters(t2)
