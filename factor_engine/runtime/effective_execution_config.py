"""Validated public fast-path choices, not a new resource authority."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EffectiveExecutionConfig:
    requested_parallel: str
    parallel: str
    scheduler: str
    resource_profile: str
    auto_shard: bool
    native_fusion: bool
    result_policy: str
    write_results: bool
    retain_results: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_fast_execution_config(*, scheduler: str = "adaptive", parallel: str = "hybrid",
                                  resource_profile: str = "balanced", auto_shard: bool = True,
                                  native_fusion: bool = True, result_policy: str = "sink",
                                  write_results: bool = True) -> EffectiveExecutionConfig:
    """Validate before planning or writes. Caller must apply every effective field.

    Shared Python context/cache closures currently require threads. Auto/hybrid
    are requests resolved to threads, never claims that process work was used.
    Broker decisions still control thread and memory admission.
    """
    for name, value in (("auto_shard", auto_shard), ("native_fusion", native_fusion),
                        ("write_results", write_results)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be bool")
    if scheduler != "adaptive":
        raise ValueError("fast scheduler supports only 'adaptive'")
    if parallel not in {"thread", "hybrid", "auto"}:
        raise ValueError("fast parallel supports thread/auto/hybrid resolved to thread; process unsupported")
    if resource_profile != "balanced":
        raise ValueError("fast resource_profile supports only 'balanced' with the existing broker policy")
    if not auto_shard:
        raise ValueError("auto_shard=False unsupported: resource/OOM replanning remains governed by scheduler")
    if result_policy not in {"sink", "return"}:
        raise ValueError("fast result_policy supports 'sink' or compute-only 'return'")
    if write_results and result_policy == "return":
        raise ValueError("result_policy='return' requires write_results=False; use sink for materialization")
    return EffectiveExecutionConfig(
        requested_parallel=parallel, parallel="thread", scheduler=scheduler,
        resource_profile=resource_profile, auto_shard=auto_shard, native_fusion=native_fusion,
        result_policy=result_policy, write_results=write_results,
        retain_results=not write_results and result_policy == "return",
    )
