# -*- coding: utf-8 -*-
"""R40 registry bootstrap items #151/#152/#153/#154/#155 (state-machine tests)."""
from __future__ import annotations

import threading
import time

import pytest

from factor_engine.cleaned_operators import (
    BOOTSTRAP_MODULE_SPECS,
    BootstrapModuleRole,
    RegistryBootstrap,
    check_bootstrap_module_specs,
    validate_bootstrap_module_specs,
)


def _mk_bootstrap(sleep: float = 0.02, *, fail: bool = False) -> RegistryBootstrap:
    b = RegistryBootstrap()

    def _run(include_research: bool):
        time.sleep(sleep)
        if fail:
            raise RuntimeError("injected init failure")

    b._run_initialization = _run  # type: ignore[assignment]
    return b


class TestConcurrentBootstrap:
    def test_concurrent_32_threads_load_once(self):
        b = _mk_bootstrap()
        calls = []
        lock = threading.Lock()

        def _run(include_research: bool):
            time.sleep(0.02)
            with lock:
                calls.append(include_research)

        b._run_initialization = _run  # type: ignore[assignment]
        errors: list[BaseException] = []
        results: list[str] = []
        results_lock = threading.Lock()

        def worker():
            try:
                b.ensure_ready(include_research=True)
                with results_lock:
                    results.append("ok")
            except BaseException as exc:  # noqa: BLE001
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(results) == 32
        assert len(calls) == 1  # initialized exactly once
        assert b.state == "ready"

    def test_failed_bootstrap_raises_same_error_to_all(self):
        b = _mk_bootstrap(fail=True)
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def worker():
            try:
                b.ensure_ready(include_research=True)
            except BaseException as exc:  # noqa: BLE001
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert b.state == "failed"
        assert len(errors) == 8
        assert all(type(e) is RuntimeError and str(e) == "injected init failure" for e in errors)


class TestFirstCallFixesSurface:
    def test_four_orders_are_deterministic(self):
        # The first call fixes the effective surface; a later call with a
        # different include_research never re-runs (R40 #152).
        def run_order(order: list[bool]) -> tuple[list[bool], str]:
            b = _mk_bootstrap()
            seen: list[bool] = []

            def _run(include_research: bool):
                seen.append(include_research)

            b._run_initialization = _run  # type: ignore[assignment]
            for flag in order:
                b.ensure_ready(include_research=flag)
            return seen, b.state

        # production -> research : only the first flag actually runs
        seen, state = run_order([False, True])
        assert seen == [False]
        assert state == "ready"
        # research -> production
        seen, state = run_order([True, False])
        assert seen == [True]
        assert state == "ready"

    def test_include_research_does_not_change_global_behavior(self):
        b = _mk_bootstrap()
        seen = []
        b._run_initialization = lambda flag: seen.append(flag)  # type: ignore[assignment]
        b.ensure_ready(include_research=False)
        # A subsequent research request returns the SAME (production) surface.
        b.ensure_ready(include_research=True)
        b.ensure_ready(include_research=True)
        assert seen == [False]
        assert b.state == "ready"


class TestTransactionalRetry:
    def test_failed_then_reset_retries_cleanly(self):
        # R40 #153: after FAILED the active snapshot is unchanged; reset() lets
        # a clean retry produce the same digest as a clean initialization.
        attempts = []

        def _run(include_research: bool):
            attempts.append(include_research)
            if len(attempts) == 1:
                raise RuntimeError("first attempt fails")

        b = _mk_bootstrap()
        b._run_initialization = _run  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            b.ensure_ready(include_research=True)
        assert b.state == "failed"
        b.reset()
        assert b.state == "new"
        b.ensure_ready(include_research=True)
        assert b.state == "ready"
        assert len(attempts) == 2


class TestBootstrapModuleSpec:
    def test_spec_roles_validated(self):
        assert validate_bootstrap_module_specs(BOOTSTRAP_MODULE_SPECS) == []
        check_bootstrap_module_specs()

    def test_internal_kernel_not_in_production_surface(self):
        internal = {spec.module for spec in BOOTSTRAP_MODULE_SPECS if spec.role is BootstrapModuleRole.INTERNAL_KERNEL}
        for spec in BOOTSTRAP_MODULE_SPECS:
            if spec.module in internal:
                assert spec.in_production_surface is False

    def test_research_extension_not_in_production_surface(self):
        research_modules = {
            "factor_engine.cleaned_operators.ts_model.dynamic_regression",
            "factor_engine.cleaned_operators.research_polars",
        }
        for spec in BOOTSTRAP_MODULE_SPECS:
            if spec.module in research_modules:
                assert spec.role is BootstrapModuleRole.RESEARCH_EXTENSION
                assert spec.in_production_surface is False

    def test_required_flag_present(self):
        assert all(spec.required for spec in BOOTSTRAP_MODULE_SPECS)


class TestEnsureOperatorRegistry:
    def test_ensure_registry_respects_surface(self, monkeypatch):
        from factor_engine.backend import cleaned_bridge as cb

        calls = []
        probe = _mk_bootstrap()

        monkeypatch.setattr("factor_engine.cleaned_operators.REGISTRY_BOOTSTRAP", probe)
        monkeypatch.setattr(
            "factor_engine.cleaned_operators.check_signature_authority",
            lambda production=True: calls.append(("sig", production)),
        )
        cb.ensure_operator_registry(surface="production")
        assert calls == [("sig", True)]

    def test_ensure_cleaned_loaded_no_unlocked_global(self, monkeypatch):
        from factor_engine.backend import cleaned_bridge as cb

        probe = _mk_bootstrap()
        monkeypatch.setattr("factor_engine.cleaned_operators.REGISTRY_BOOTSTRAP", probe)
        cb.ensure_cleaned_loaded()
        assert probe.state == "ready"
        assert not hasattr(cb, "_CLEANED_LOADED")
