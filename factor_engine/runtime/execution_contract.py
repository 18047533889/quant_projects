# -*- coding: utf-8 -*-
"""ExecutionContract + HistoryRequirement — the SINGLE authority for
statefulness / chunking / checkpoint / lookback-warmup (WS-D findings
#256-#258).

Every layer that used to consult a scattered stateful-name list or a
``_LOOKBACK_PARAM_PRIORITY`` name-guessing tuple now routes through here:

* ``execution_contract(canonical)``  — state_model / chunking / checkpoint_schema.
* ``history_requirement(canonical, params)`` — the ONE lookback authority,
  returning a ``HistoryRequirement`` instead of an integer sentinel.
* ``factor_history_requirement(ir)`` — combined requirement for a whole factor IR.

The legacy ``FULL_HISTORY_LOOKBACK_SENTINEL`` integer survives ONLY as the
serialized-analysis encoding that ``ir.analyzer`` writes for ``requires_full_history``
factors; it is no longer the authority for anything.  ``warmup_service``,
``composite_lowering`` and the analyzer integration read the dataclasses below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Serialized-analysis encoding only.  ``analysis.lookback`` carries this value
# when ``requires_full_history`` was True at analysis time (see
# ``cleaned_operators.production_policy_extensions_v2``).  Consumers must treat
# it as a compatibility signal, never as a window size.
FULL_HISTORY_LOOKBACK_SENTINEL = 1_000_000_000

# The stateful seed (WS-D #256).  This replaces ``contract_hardening._STATEFUL_CANONICALS``
# and any ``_STATE_MODEL_*`` lists: nothing else may be consulted for statefulness /
# chunking.  The segmented (checkpoint-capable) members are registered in
# ``stateful_contract.StatefulCheckpointRegistry``; the rest are recursive /
# full-history-replay operators with no checkpoint-restore.
_STATEFUL_CANONICALS: frozenset[str] = frozenset({
    # --- checkpoint-capable (segmented) recursive kernels -------------------
    "ts_ema",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_ewm_cov",
    "ts_ewm_corr",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    # --- recursive stateful operators without a checkpoint restore -----------
    # (from contract_hardening's list; ts_max_run / ts_min_run are NOT present
    # in the registry as of round-7, so they are intentionally absent here).
    "ts_sma_cn",
    "MACD",
    "KAMA",
    "Supertrend",
    "SupertrendDirection",
    "PSAR",
    "DMI_plus",
    "DMI_minus",
    "DX",
    "NATR",
    "PPO",
    "PPO_signal",
    "PPO_hist",
    "PVO",
    "PVO_signal",
    "PVO_hist",
    "KeltnerMid",
    "KeltnerUpper",
    "KeltnerLower",
    "KeltnerPosition",
    "TSI",
    "TSI_signal",
    "DEMA",
    "TEMA",
    "ChaikinOscillator",
    "ForceIndex",
    "expanding_rank",
    "hump_decay",
    "state_latch",
    "state_hold",
    "state_slew_limit",
    "state_deadband",
    "state_ewm_if",
    "state_since_reduce",
    "event_refractory",
    "cross_event",
    "directional_change_state",
    "directional_change_extent",
    "state_since_trend_tstat",
    "ts_cusum_break_score",
    "ts_threshold_cycle_period",
    "ts_threshold_cycle_asymmetry",
    "state_episode_mfe",
    "state_episode_mae",
    "state_episode_efficiency",
    "state_episode_retrace_ratio",
    "state_episode_excursion_balance",
    "ts_interval_nesting_depth",
    "candle_gap_atr",
})

_MACD_FAMILY = frozenset({"MACD", "MACD_line", "MACD_signal", "MACD_hist"})

# Window-like parameter names used ONLY to derive a finite warmup from an
# operator's OWN declared ``param_names`` (never applied to un-declared names
# and never a substitute for a declared history contract).  Restricted to the
# operator's declared parameters inside ``_declared_window_rows``.
_WINDOW_LIKE_PARAM_NAMES = frozenset({
    "window", "span", "period", "periods", "d", "n", "lag", "lookback",
    "max_lookback", "fast", "slow", "signal", "fast_period", "slow_period",
    "signal_period", "signal_span", "fast_window", "slow_window",
    "signal_window", "short_window", "medium_window", "long_window",
    "ema_window", "atr_window", "er_window", "vol_window",
    "tenkan_window", "kijun_window", "senkou_b_window",
    "baseline_window", "price_window", "volume_window", "turnover_window",
})


@dataclass(frozen=True)
class ExecutionContract:
    """Three-axis execution contract for one canonical (WS-D #256).

    Attributes:
        state_model: ``stateless`` | ``recursive`` | ``episode`` | ``session_state``.
        chunking: ``independent`` | ``checkpoint`` | ``required_full_history``.
        checkpoint_schema: opaque schema id (``None`` when no checkpoint exists).
    """

    state_model: str = "stateless"
    chunking: str = "independent"
    checkpoint_schema: str | None = None

    @property
    def is_stateful(self) -> bool:
        return self.state_model != "stateless"

    @property
    def requires_full_history(self) -> bool:
        return self.chunking == "required_full_history"


@dataclass(frozen=True)
class HistoryRequirement:
    """Machine-readable lookback requirement (WS-D #257).

    ``kind`` is one of ``finite`` / ``full_history`` / ``fiscal_period`` /
    ``session``; ``rows`` is the minimum warm-up rows required before the first
    meaningful output (never the 1e9 sentinel).
    """

    kind: str = "finite"  # finite | full_history | fiscal_period | session
    rows: int = 2

    @property
    def is_full_history(self) -> bool:
        return self.kind == "full_history"


def _resolve(canonical: str) -> str:
    """Resolve DSL alias -> canonical (lazy; safe before the registry is built)."""
    if not canonical:
        return canonical
    try:
        from cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(canonical)
    except Exception:
        return canonical


def _metadata(canonical: str) -> Any | None:
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        return getattr(op, "metadata", None)
    except Exception:
        return None


def _checkpoint_spec(canonical: str) -> Any | None:
    try:
        from stateful_contract import StatefulCheckpointRegistry

        return StatefulCheckpointRegistry.get(canonical)
    except Exception:
        return None


def execution_contract(canonical: str) -> ExecutionContract:
    """Return the single authority execution contract for ``canonical``.

    Resolution order: the checkpoint registry decides ``checkpoint`` vs
    ``required_full_history``; the stateful seed covers operators without a
    checkpoint spec; everything else is stateless/independent.
    """
    resolved = _resolve(canonical)
    spec = _checkpoint_spec(resolved)
    if spec is not None:
        if spec.segmented_execution_supported:
            return ExecutionContract(
                state_model="recursive",
                chunking="checkpoint",
                checkpoint_schema=spec.state_schema_version,
            )
        return ExecutionContract(
            state_model="recursive",
            chunking="required_full_history",
            checkpoint_schema=spec.state_schema_version,
        )
    if resolved in _STATEFUL_CANONICALS:
        return ExecutionContract(
            state_model="recursive",
            chunking="required_full_history",
            checkpoint_schema=None,
        )
    return ExecutionContract(state_model="stateless", chunking="independent")


def _bound_param(
    params: Mapping[str, Any],
    name: str,
    specs: Mapping[str, Any],
) -> int | None:
    """Resolve one parameter to a positive int from bound params, else its
    declared ParamSpec default/min, else None."""
    if name in params and params.get(name) is not None:
        value = params[name]
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    spec = specs.get(name)
    if spec is not None:
        for candidate in (getattr(spec, "default", None), getattr(spec, "min", None)):
            if candidate is None or isinstance(candidate, bool):
                continue
            try:
                parsed = int(candidate)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _declared_window_rows(canonical: str, params: Mapping[str, Any]) -> int:
    """Derive a finite warm-up row estimate from the operator's OWN declared
    history contract (ParamSpec defaults + declared window params).

    This is deliberately NOT a global ``_LOOKBACK_PARAM_PRIORITY`` name-guessing
    tuple: only parameters the operator declares in its ``param_names`` are
    considered, and each is bounded by its declared ParamSpec default/min.
    """
    meta = _metadata(canonical)
    if meta is None:
        return 0
    specs = getattr(meta, "param_specs", None) or {}
    param_names = tuple(getattr(meta, "param_names", None) or ())
    rows = 0
    for name in param_names:
        if name not in _WINDOW_LIKE_PARAM_NAMES:
            continue
        value = _bound_param(params, name, specs)
        if value is not None:
            rows = max(rows, value - 1)
    if canonical in _MACD_FAMILY:
        fast = _bound_param(params, "fast", specs) or 12
        slow = _bound_param(params, "slow", specs) or 26
        signal = _bound_param(params, "signal", specs) or 9
        rows = max(rows, max(fast, slow) - 1 + max(signal - 1, 0))
    return rows


def _minimum_warmup_rows(canonical: str, params: Mapping[str, Any] | None) -> int:
    spec = _checkpoint_spec(canonical)
    min_rows = int(getattr(spec, "minimum_history", 0) or 0) if spec is not None else 0
    declared = _declared_window_rows(canonical, params or {})
    return max(2, min_rows, declared)


def history_requirement(
    canonical: str,
    params: Mapping[str, Any] | None = None,
) -> HistoryRequirement:
    """THE one lookback authority.

    Returns a ``HistoryRequirement`` for a canonical — never a raw integer and
    never the 1e9 sentinel.  Recursive / full-history-replay operators report
    ``kind='full_history'``; everything else reports a finite row count derived
    from the operator's declared history contract (safe default: finite, 2 rows).
    """
    resolved = _resolve(canonical)
    contract = execution_contract(resolved)
    rows = _minimum_warmup_rows(resolved, params)
    if contract.requires_full_history:
        return HistoryRequirement(kind="full_history", rows=rows)
    return HistoryRequirement(kind="finite", rows=rows)


def _node_params(node: Any) -> dict[str, Any]:
    """Extract bound operator params from an IRNode (kwargs attrs + positional
    literal inputs mapped through the operator's declared ``param_names``)."""
    params: dict[str, Any] = {}
    if getattr(node, "attrs", None):
        params.update(dict(node.attrs))
    meta = _metadata(getattr(node, "op", ""))
    if meta is None:
        return params
    param_names = tuple(getattr(meta, "param_names", None) or ())
    for index, child in enumerate(getattr(node, "inputs", ()) or ()):
        if getattr(child, "op", None) == "literal":
            name = param_names[index] if index < len(param_names) else None
            if name:
                params.setdefault(name, getattr(child, "attrs", {}).get("value"))
    return params


def factor_history_requirement(ir: Any | None) -> HistoryRequirement:
    """Combined history requirement for a whole factor IR (max rows; full-history
    wins).  Mirrors what the analyzer derives so warmup/planner agree."""
    combined = HistoryRequirement(kind="finite", rows=2)
    if ir is None:
        return combined

    def walk(node: Any) -> None:
        nonlocal combined
        if node is None:
            return
        op = getattr(node, "op", None)
        if op:
            requirement = history_requirement(str(op), _node_params(node))
            rows = max(combined.rows, requirement.rows)
            if requirement.is_full_history or combined.is_full_history:
                combined = HistoryRequirement(kind="full_history", rows=rows)
            else:
                combined = HistoryRequirement(kind="finite", rows=rows)
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(ir)
    return combined


def is_full_history_lookback(lookback: int) -> bool:
    """True when a raw ``analysis.lookback`` encodes full-history replay via the
    legacy serialized sentinel.  Kept only for serialized-analysis compatibility;
    the authority is ``history_requirement`` / ``factor_history_requirement``."""
    try:
        return int(lookback) >= FULL_HISTORY_LOOKBACK_SENTINEL
    except (TypeError, ValueError):
        return False


__all__ = [
    "ExecutionContract",
    "HistoryRequirement",
    "FULL_HISTORY_LOOKBACK_SENTINEL",
    "execution_contract",
    "factor_history_requirement",
    "history_requirement",
    "is_full_history_lookback",
]
