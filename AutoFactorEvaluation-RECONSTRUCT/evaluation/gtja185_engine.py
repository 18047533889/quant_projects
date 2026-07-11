"""Efficient GTJA185 preflight and production-aware FactorEngine execution."""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import pandas as pd

from factor_packs.gtja185 import FactorDefinition
from integrations.quant_platform import InMemoryFrameSource, bootstrap_quant_platform

bootstrap_quant_platform()

from api.dsl_parser import parse_factor  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from runtime.pit_audit import assert_pit_safe  # noqa: E402


def compile_pack_once(
    factors: Sequence[FactorDefinition],
    frame: pd.DataFrame,
    *,
    backend: str,
    run_mode: str,
) -> tuple[FactorEngine, list[Any], dict[str, float]]:
    """Perform parser/analyzer/PIT preflight without compiling plans twice.

    ``FactorEngine.run_many`` performs the authoritative logical/physical compile,
    production-policy checks and execution. Preflight only lowers the expression to IR
    and audits causality, so each factor reaches the full compiler exactly once.
    """
    source = InMemoryFrameSource(
        frame,
        time_column="datetime",
        instrument_column="asset",
    )
    engine = FactorEngine(
        build_backend(backend),
        source,
        run_mode=run_mode,
    )
    parsed: list[Any] = []
    timings: dict[str, float] = {}
    for definition in factors:
        factor = parse_factor(
            definition.formula,
            name=definition.name,
            freq="1d",
            universe="GTJA185",
            description=definition.description,
        )
        started = time.perf_counter()
        analysis = engine.analyzer.lower(factor.expr)
        assert_pit_safe(
            analysis.ir,
            enforce=True,
            forbid_forward_fill=True,
        )
        timings[definition.name] = time.perf_counter() - started
        parsed.append(factor)
    return engine, parsed, timings


def _coerce_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        result = value.rename(name)
    elif isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        result = value.iloc[:, 0].rename(name)
    elif hasattr(value, "to_pandas"):
        converted = value.to_pandas()
        if not isinstance(converted, pd.Series):
            raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
        result = converted.rename(name)
    else:
        raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
    if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels != 2:
        raise ValueError(f"factor {name} must return MultiIndex(datetime, asset)")
    result.index = result.index.set_names(["datetime", "asset"])
    if result.index.has_duplicates:
        raise ValueError(f"factor {name} returned duplicate (datetime, asset) keys")
    return pd.to_numeric(result, errors="coerce").sort_index()


def execute_pack_once(
    engine: FactorEngine,
    factors: Sequence[Any],
    *,
    batch_size: int,
    market: str,
) -> tuple[dict[str, pd.Series], dict[str, float], dict[str, str]]:
    """Compile/execute each batch once and enable hard production run flags."""
    results: dict[str, pd.Series] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    production = str(getattr(engine, "run_mode", "research")).lower() == "production"
    run_flags = {
        "input_dq_check": production,
        "input_dq_strict": True,
        "auto_warmup": production,
        "trim_warmup": True,
        "market": market,
        "pit_enforce": production,
        "pit_forbid_forward_fill": production,
    }
    for offset in range(0, len(factors), batch_size):
        batch = list(factors[offset : offset + batch_size])
        started = time.perf_counter()
        try:
            output = engine.run_many(
                batch,
                enable_cse=True,
                **run_flags,
            )
            per_factor = (time.perf_counter() - started) / max(len(batch), 1)
            for factor in batch:
                results[factor.name] = _coerce_series(
                    output["results"][factor.name],
                    factor.name,
                )
                timings[factor.name] = per_factor
        except Exception as batch_error:
            for factor in batch:
                one_started = time.perf_counter()
                try:
                    output = engine.run(factor, **run_flags)
                    results[factor.name] = _coerce_series(
                        output["result"],
                        factor.name,
                    )
                    timings[factor.name] = time.perf_counter() - one_started
                except Exception as factor_error:
                    errors[factor.name] = (
                        f"batch={type(batch_error).__name__}: {batch_error}; "
                        f"single={type(factor_error).__name__}: {factor_error}"
                    )
    return results, timings, errors
