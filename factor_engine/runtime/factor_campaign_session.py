# -*- coding: utf-8
"""R39 PERF-070: ``FactorCampaignSession`` — one shared engine per identical identity.

A campaign session groups the factors of an identical immutable
``ExecutionIdentity`` behind ONE engine + source session.  The per-factor
``materialize_incremental`` still runs on the shared engine; factor-level
semantic identity is still validated per factor by the engine.  Different
source configs (a different identity) never share a session/engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.execution_identity import ExecutionIdentity


@dataclass
class CampaignFactor:
    """One factor inside a campaign: its loaded config, factor, path and opts."""

    config: Any
    factor: Any
    path: str
    opts: Any

    @property
    def name(self) -> str:
        return self.factor.name

    def to_incremental_materialize_kwargs(self) -> dict[str, Any]:
        """``FactorEngine.materialize_incremental(**kwargs)`` for this factor."""
        return self.opts.to_incremental_materialize_kwargs()


@dataclass
class FactorCampaignSession:
    """A batch of factors sharing an identical ``ExecutionIdentity`` and one engine.

    Attributes:
        identity: the strict immutable identity the whole campaign shares.
        engine: the single engine reused across the campaign.
        source_session: the shared source/read session (``engine.data_source``
            when not supplied).
        compiled_factors: the per-factor records (config / factor / path / opts).

    The engine exposes CSE-aware batch compile via ``compile_many``; this is
    the shared compiled artifact held by the session and cached across calls.
    """

    identity: ExecutionIdentity
    engine: Any
    source_session: Any = None
    compiled_factors: list[CampaignFactor] = field(default_factory=list)
    _compile_result: dict[str, Any] | None = field(default=None, repr=False)
    _compile_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.source_session is None:
            self.source_session = getattr(self.engine, "data_source", None)

    @property
    def factor_names(self) -> list[str]:
        """Names of the factors in this campaign (order preserved)."""
        return [rec.name for rec in self.compiled_factors]

    @property
    def compile_count(self) -> int:
        """Number of actual batch-compile passes performed (0 until first call)."""
        return self._compile_count

    def add_factor(
        self,
        config: Any,
        factor: Any,
        path: str,
        opts: Any,
    ) -> None:
        """Register one factor into this campaign (order preserved)."""
        self.compiled_factors.append(CampaignFactor(config, factor, str(path), opts))

    def compile(self, *, enable_cse: bool = True) -> dict[str, Any]:
        """Batch-compile all factors through the shared engine (CSE-aware).

        Uses the engine's exposed ``compile_many`` so shared sub-expressions
        across factors are hoisted once for the whole campaign.  The compiled
        ``DAGPlan`` is cached; repeated calls return the cached artifact without
        recompiling (``compile_count`` stays 1).

        Returns:
            ``{"plan": DAGPlan, "factors": [Factor, ...]}``.
        """
        if self._compile_result is not None:
            return self._compile_result
        if not self.compiled_factors:
            self._compile_result = {"plan": None, "factors": []}
            return self._compile_result
        self._compile_count += 1
        factors = [rec.factor for rec in self.compiled_factors]
        plan = self.engine.compile_many(factors, enable_cse=enable_cse)
        self._compile_result = {"plan": plan, "factors": factors}
        return self._compile_result
