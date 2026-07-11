"""AutoFactorEvaluation evaluation package.

The GTJA185 batch evaluator is dependency-light and must remain importable without
loading the legacy timeseries/indicator/label stack.  Legacy public names are
therefore resolved lazily; importing ``evaluation.gtja185_batch`` no longer
implicitly imports vectorbt or opens legacy evaluation resources.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EvaluationPipelineResult": ("evaluation.pipeline", "EvaluationPipelineResult"),
    "run_evaluation_pipeline": ("evaluation.pipeline", "run_evaluation_pipeline"),
    "process_factor_dir": ("evaluation.scripts.worker", "process_factor_dir"),
    "EvaluationRouter": ("evaluation.scripts.router", "EvaluationRouter"),
    "EvaluationConfig": ("evaluation.config", "EvaluationConfig"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
