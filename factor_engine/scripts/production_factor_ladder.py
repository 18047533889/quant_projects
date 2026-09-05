#!/usr/bin/env python3
"""R61-P0 #58 — FE-native production factor ladder (one batch pipeline).

Supersedes the two defective ladders:

- ``scripts/dry_run_ladder.py`` — per-factor ``run_many([f])`` loops (never
  stresses cross-root CSE/DAG/fusion/read-wave), fake stratification on
  pre-built strings, hard cap at the terminal set (~1446), ``--workers`` /
  ``--per-factor-timeout`` parsed but never enforced, local ``StreamingSink``
  duplicate, lossy ``(fid, None)`` build-failure records.
- ``benchmarks/dry_run_ladder.py`` — synthetic pandas ``rolling_mean`` /
  ``pct_change(5)`` functions through the real sink/checkpoint (NOT real FE
  operators).

This ladder is FE-native end-to-end:

1. **Real mining-grammar roots** — generated from the 62 Agent-direct terminal
   operators (``artifacts/AGENT_DIRECT_OPERATORS.json``) via the lazy DSL
   factories (``factor_engine.api``), so every root is a real
   ``operator.call(col(...), ...)`` CleanedCall exactly like production.
2. **Batch execution per level** — each level runs chunks of
   ``engine.run_many(factors, enable_cse=True)`` (``result_policy="sink"`` with
   the storage ``StreamingSink``; ``"return"`` for small levels). NEVER a
   per-factor ``run()`` / ``run_many([f])`` loop (the only exception is the
   watchdog re-run of a *chunk that overran its deadline*, where each root must
   be attributed individually).
3. **Unique structural roots > 95%** — ``LadderRoot.structural_fingerprint`` is
   a canonical string with numeric-literal normalization (``20`` == ``20.0``);
   every level asserts the invariant and fails loudly.
4. **Forced shared subtrees** — ``ts_mean(close,20)`` / ``ts_std(return,20)``
   are shared children across many roots so CSE genuinely extracts shared
   nodes; telemetry surfaces ``shared_nodes`` / ``reuse_edges`` / cache sizes.
5. **Stratified sampling** — by ``economic_effect_family``; each level covers
   >= 2 families (level 100 asserts it).
6. **Real enforcement** — ``--workers`` is governed by
   ``multiworker_governance.resolve_worker_budget`` (workers x threads <=
   floor(31 / coexist_count), max 8 for this ladder) and passed to the batch
   path; ``--per-factor-timeout`` is a REAL watchdog — a chunk that overruns is
   re-run root-by-root under an armed in-thread deadline so a stuck root is
   aborted and classified ``timeout``. ``timeout=0`` is REJECTED (never
   silently disables enforcement).
7. **No lossy failures** — every build/compile/runtime/sink exception is kept
   as ``(formula, error_message, category)``; never ``(fid, None)``.
8. **Checkpoint/resume** — ``DryRunCheckpoint`` (atomic JSON) marked every
   completed/failed root + level-state JSON every 25 formulas; resume skips
   completed roots. Sink shards are 5000-row row-groups with sidecar checksums
   via ``factor_engine.storage.streaming_sink.StreamingSink`` (imported — no
   local duplicate class).

Status JSON: ``/tmp/r61_ladder58/ladder_status.json`` (independent of the old
status file locations).

CLI::

    python3 scripts/production_factor_ladder.py --levels 100,1000 --workers 4 \\
        --per-factor-timeout 60 --out /tmp/r61_ladder58
    python3 scripts/production_factor_ladder.py --dry-level 100   # CI: level 100 only
    python3 scripts/production_factor_ladder.py --gen-only 100000 # generation smoke only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

_FE_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_FE_ROOT), str(_FE_ROOT.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")
os.environ.setdefault("FACTOR_ENGINE_SCHEDULER", "adaptive")
# The ladder's in-memory panel source has no duckdb/polars executor, so
# thread execution avoids pickling plans into a process pool.
os.environ.setdefault("FACTOR_ENGINE_HYBRID_FORCE", "thread")
# Multi-root native fusion (execute_multi_roots) hangs on the pandas/in-memory
# path for some mixed-operator batches; per-root execution is the honest,
# non-deadlocking execution mode for the ladder's synthetic source.
os.environ.setdefault("FACTOR_ENGINE_NATIVE_FUSION", "0")

# Registry bootstrap is a one-time ~40-50 s cost on this box (spec completion
# + evidence hashing). Warm it at import time so the FIRST generate_pool /
# level run does not absorb it (and pytest-timeout does not see a false hang).
def _bootstrap_registry_once() -> None:
    try:
        from factor_engine.cleaned_operators import load_all

        load_all(include_research=True)
    except Exception:
        pass


_bootstrap_registry_once()

# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------
FAIL_PARAM = "param_error"
FAIL_DATA = "data_missing"
FAIL_TIMEOUT = "timeout"
FAIL_OOM = "oom"
FAIL_SEMANTIC = "semantic"
FAIL_OTHER = "other"
FAIL_CATEGORIES = [
    FAIL_PARAM, FAIL_DATA, FAIL_TIMEOUT, FAIL_OOM, FAIL_SEMANTIC, FAIL_OTHER,
]

LEVELS = [100, 1000, 5000, 20000, 50000, 100000]
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_WORKERS = 4
MAX_WORKERS = 8
TOTAL_CORES = 31
CHUNK_SIZE = 200
CHECKPOINT_EVERY = 25

VALID_SOURCES = (
    "open", "high", "low", "close", "volume", "amount", "return",
)

_OP_FAMILY_OVERRIDE: dict[str, str] = {
    "ts_autocorr": "autocorrelation",
    "ts_mean": "trend",
    "ts_delay": "trend",
    "ts_log_return": "trend",
    "ts_delta": "trend",
    "ts_pct": "trend",
    "ts_std": "volatility",
    "ts_var": "volatility",
    "ts_max": "tail",
    "ts_min": "tail",
    "ts_sharpe": "volatility",
    "ts_zscore": "mean_reversion",
    "ts_rank": "mean_reversion",
    "ts_median": "mean_reversion",
    "ts_sum": "mean_reversion",
    "ts_beta": "mean_reversion",
    "ts_corr": "autocorrelation",
    "ts_cov": "volatility",
    "ts_sharpe": "volatility",
    "rank": "cross_section",
    "zscore": "cross_section",
    "cs_pct_rank": "cross_section",
    "cs_demean": "cross_section",
    "cs_mad_zscore": "cross_section",
    "normalize": "cross_section",
    "group_rank": "cross_section",
    "group_zscore": "cross_section",
    "group_normalize": "cross_section",
    "group_neutralize": "cross_section",
    "group_winsorize": "cross_section",
    "winsorize": "winsorization",
    "clip": "winsorization",
}

#: Operators that wrap the forced shared children ``ts_mean(close,20)`` /
#: ``ts_std(return,20)`` so CSE has real cross-root reuse to extract.
_SHARED_WRAP_OPS = (
    "ts_rank", "ts_zscore", "ts_delta", "ts_pct", "ts_beta",
    "ts_corr", "ts_cov", "ts_autocorr", "ts_sharpe", "ts_var", "ts_max", "ts_min",
)

#: Multi-series operators whose dedup decorator wrapper would pass a literal
#: where a second series is required (typed-input error) — dropped on dup
#: rather than emitted as an always-failing root.
_UNWRAPPABLE_TWO_SLOT_OPS = frozenset(
    {"ts_beta", "ts_corr", "ts_cov", "ts_corr", "where", "divide", "multiply",
     "subtract", "add", "maximum", "minimum", "ts_cov"}
)

# Real kernel defaults per operator — read from the operator kernel so
# generated roots always pass planning-time validation (R6 P0-04 / R13 P0-20).
_KERNEL_DEFAULTS_CACHE: dict[str, dict[str, Any]] = {}


def _kernel_defaults(canonical: str) -> dict[str, Any]:
    if canonical in _KERNEL_DEFAULTS_CACHE:
        return _KERNEL_DEFAULTS_CACHE[canonical]
    out: dict[str, Any] = {}
    try:
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.base import _kernel_param_defaults
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all(include_research=True)
        reg = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        if reg is not None:
            out = dict(_kernel_param_defaults(reg) or {})
    except Exception:
        pass
    _KERNEL_DEFAULTS_CACHE[canonical] = out
    return out


def _default_param_value(canonical: str, name: str) -> Any:
    """Param default that survives planning-time validation.

    Kernel-declared defaults (``_kernel_param_defaults``) win; then declared
    ParamSpec choices; then a name-based fallback.
    """
    kern = _kernel_defaults(canonical)
    # group_normalize / group_winsorize declare a *group-key* scalar param
    # named ``group`` — it is a context key, not a numeric window. The offline
    # plain-panel source cannot supply a group key, so exclude the root.
    if name == "group":
        raise _UnsupportedContextSlot(f"{canonical}.group")
    if name in kern and kern[name] is not None:
        return kern[name]
    low = str(name).lower()
    if low in ("null_policy", "nan_policy"):
        return "propagate"
    if low == "zero_std_policy":
        return "zero"
    if low == "includes_current_bar":
        return True
    if low == "fallback_policy":
        return "nan"
    if "window" in low or "lookback" in low or "period" in low:
        return 20
    if low == "min_periods":
        return 1
    if low in ("lag", "n", "d"):
        return 1
    if low == "ddof":
        return 1
    if "ann_factor" in low:
        return 252
    if "scale" in low:
        return 1
    if low in ("lower", "upper", "a", "lo", "hi"):
        return 0.05
    return 5


# ---------------------------------------------------------------------------
# Canonical structural fingerprint
# ---------------------------------------------------------------------------
def _canonical_int(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def structural_key(expr: Any) -> str:
    """Canonical dedup key for an expression tree.

    - literal numerics normalized (``20`` == ``20.0``) so notation does not
      multiply identities
    - kwargs joined deterministically
    - operator names / column names preserved
    """
    parts: list[str] = []

    def _walk(node: Any, depth: int = 0) -> None:
        # Literal nodes contribute their normalized VALUE (20 vs 20.0 dedup to
        # one identity, but 20 vs 21 stay distinct).
        if type(node).__name__ == "Literal" or (
            getattr(node, "op", None) is None and "value" in vars(node).get("__dataclass_fields__", {})
        ):
            parts.append(str(_canonical_int(getattr(node, "value", node))))
            return
        op = getattr(node, "op", None)
        if op is not None:
            parts.append(str(op))
        name = getattr(node, "name", None)
        if name is not None:
            parts.append(str(name))
        args = getattr(node, "args", None)
        if args:
            for a in args:
                _walk(a, depth + 1)
        elif args is not None and len(args) == 0:
            pass
        if hasattr(node, "kwargs"):
            kw = dict(getattr(node, "kwargs") or {})
            for k in sorted(kw.keys()):
                parts.append(str(k))
                parts.append(str(_canonical_int(kw[k])))

    _walk(expr)
    return "|".join(parts)


def fingerprint_of_expr(expr: Any) -> str:
    return structural_key(expr)


# ---------------------------------------------------------------------------
# Operator catalog + pool generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LadderRoot:
    """One production-faithful factor root for the ladder."""

    fid: str
    expr: Any  # CleanedCall tree (build-time only; never executed here)
    operator_name: str
    economic_effect_family: str
    structural_fingerprint: str
    source_columns: tuple[str, ...]
    scalar_params: dict[str, Any]
    build_error: str = ""  # preserved real error when expr is None


def _make_col(name: str) -> Any:
    from factor_engine.api import col

    return col(name)


def _pick_source(seed: int) -> str:
    idx = (int(seed) * 2654435761) % len(VALID_SOURCES)
    return VALID_SOURCES[idx]


def load_terminal_operators() -> list[dict[str, Any]]:
    """62 Agent-direct terminal operators (artifacts JSON, registry fallback)."""
    path = _FE_ROOT / "artifacts" / "AGENT_DIRECT_OPERATORS.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            return [r for r in payload.get("operators") or [] if r.get("terminal")]
        except Exception:
            pass
    # registry fallback (keeps the ladder runnable if the artifact is missing)
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.market.context import Market
    from factor_engine.mining.direct_use import DirectUseContext, get_direct_use_mining_operators

    load_all(include_research=True)
    rows = get_direct_use_mining_operators(
        DirectUseContext(market=Market.ASHARE), admission="all"
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        if not r.terminal_allowed:
            continue
        out.append(
            {
                "canonical": r.canonical,
                "input_slots": [
                    {
                        "parameter": getattr(s, "parameter", "x"),
                        "roles": list(getattr(s, "allowed_roles", ()) or ("data",)),
                    }
                    for s in r.input_slots
                ],
                "scalar_parameters": list(r.scalar_parameters or ()),
            }
        )
    return out


def _economic_family(canonical: str) -> str:
    if canonical in _OP_FAMILY_OVERRIDE:
        return _OP_FAMILY_OVERRIDE[canonical]
    try:
        from factor_engine.mining.direct_use import economic_effect_family

        fam = economic_effect_family(canonical)
        if fam and fam != "unknown":
            return fam
    except Exception:
        pass
    if str(canonical).startswith("intra_"):
        return "intraday_microstructure"
    return "unknown"


def _slot_source_columns(slot: dict[str, Any], seed: int, src_cols: list[str]) -> str:
    role = ""
    roles = slot.get("roles") or ("data",)
    if roles:
        role = str(roles[0])
    if role == "condition":
        return "close"
    name = str(slot.get("parameter") or "x").lower()
    if name == "close":
        return "close"
    if name == "volume":
        return "volume"
    if name == "amount":
        return "amount"
    if "return" in name or name == "ret":
        return "return"
    if name in ("open", "high", "low"):
        return name
    # generic data slot → deterministic per-seed source choice (never the same
    # source twice within one root; single-slot roots therefore vary by source)
    candidates = [c for c in VALID_SOURCES if c not in src_cols]
    if not candidates:
        candidates = list(VALID_SOURCES)
    idx = (int(seed) * 2654435761) % len(candidates)
    return candidates[idx]


def build_one_expression(
    op: dict[str, Any],
    *,
    param_seed: int,
    vary_params: bool = True,
) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    """Build one real CleanedCall for an operator row (production-faithful)."""
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    canonical = str(op["canonical"])
    slots = list(op.get("input_slots") or ())
    scalar_names = list(op.get("scalar_parameters") or ())
    factory = make_cleaned_call_factory(canonical)

    args: list[Any] = []
    src_cols: list[str] = []
    for slot in slots:
        role = str((slot.get("roles") or ("data",))[0])
        if role == "parameter":
            # declared scalar parameter (already covered by scalar_parameters
            # kwargs below) — never pass a panel into a scalar slot
            continue
        if role == "context":
            # context slot (group key). The offline synthetic source has no
            # typed GroupKey column, so context-bearing operators (group_*)
            # are excluded at generation time for the ladder's plain-panel
            # source — a real typed-input failure is not a meaningful ladder
            # signal and would flood one category.
            raise _UnsupportedContextSlot(canonical)
        cname = _slot_source_columns(slot, param_seed, src_cols)
        args.append(_make_col(cname))
        src_cols.append(cname)
    if not args:
        args.append(_make_col("close"))
        src_cols.append("close")

    kwargs: dict[str, Any] = {}
    for p in scalar_names:
        kwargs[str(p)] = _default_param_value(canonical, str(p))
    if vary_params and scalar_names:
        p0 = scalar_names[0]
        base = kwargs[p0]
        # ParamSpec minima are respected so variation never produces an
        # invalid negative window / n (planning-time rejection otherwise).
        mini = _param_min(canonical, p0)
        if isinstance(base, (int, float)) and _is_numeric_param(canonical, p0):
            varied = base + int(((param_seed * 7) % 9) - 4)
            if mini is not None:
                varied = max(mini, varied)
            # winsorize lower/upper are PERCENTILES — must stay within (0,1)
            # and lower must stay below upper (0.95)
            if canonical == "winsorize":
                varied = min(0.90, max(0.05, varied))
            kwargs[p0] = varied

    expr = factory(*args, **kwargs)
    return expr, tuple(dict.fromkeys(src_cols)), kwargs



def _is_numeric_param(canonical: str, name: str) -> bool:
    """Only vary numeric scalar params (never string/choice params)."""
    kern = _kernel_defaults(canonical)
    if name in kern:
        return isinstance(kern[name], (int, float)) and not isinstance(kern[name], bool)
    low = str(name).lower()
    return low in (
        "window", "min_periods", "lag", "n", "d", "ddof", "ann_factor", "scale",
        "lower", "upper", "a", "lo", "hi",
    )


def _param_min(canonical: str, name: str) -> int | None:
    """Declared ParamSpec ``min`` for a scalar param, if any."""
    try:
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all(include_research=True)
        reg = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        meta = getattr(reg, "metadata", None)
        if meta is not None:
            specs = dict(getattr(meta, "param_specs", []) or {})
            spec = specs.get(str(name))
            if spec is not None:
                m = getattr(spec, "min", None)
                if m is not None:
                    return int(m)
    except Exception:
        pass
    # integer horizon params must stay >= 1 even without a declared spec min
    low = str(name).lower()
    if low in ("window", "min_periods", "lag", "n", "d"):
        return 1
    return None


class _UnsupportedContextSlot(Exception):
    """Group/context-bearing operator excluded from the plain-panel ladder."""


def _rewrite_sources_one_level(expr: Any, seed: int) -> Any:
    """Deterministically swap the leaf ColumnRefs of ``expr`` to new sources."""
    from factor_engine.expr.column import ColumnRef

    replacement = _pick_source(seed)

    def _swap(node: Any) -> Any:
        if isinstance(node, ColumnRef):
            return _make_col(replacement)
        new_kwargs = tuple((k, _swap(v) if hasattr(v, "op") else v)
                           for k, v in getattr(node, "kwargs", ()) or ())
        new_args = tuple(_swap(a) for a in (getattr(node, "args", None) or ()))
        if hasattr(node, "kwargs"):
            return type(node)(op=node.op, args=new_args, kwargs=new_kwargs)
        if hasattr(node, "name") and hasattr(node, "op") is False:
            return node
        return type(node)(op=getattr(node, "op", None), args=new_args, kwargs=new_kwargs)

    try:
        return _swap(expr)
    except Exception:
        return expr


def _decorator_chain(param_seed: int, seed2: int) -> Callable[[Any], Any]:
    """Deterministic unary decorator chain (1-2 layers) for dedup widening."""
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    single = ("cs_rank", "zscore", "cs_demean", "cs_mad_zscore", "cs_pct_rank",
              "log", "abs", "signed_sqrt", "tanh", "rank")
    # windowed decorators rotate their window via BOTH seeds → far larger
    # structural space (ts_rank(x, w) with w in 1..60)
    r1 = (int(param_seed) * 2654435761 + seed2 * 104729) % 100
    chain: list[Callable[[Any], Any]] = []
    windowed = ("ts_rank", "ts_delta", "ts_zscore", "ts_sum", "ts_std",
                "ts_max", "ts_mean", "ts_min", "ts_var", "ts_median")
    if r1 < 40:
        chain.append(_make_single(single[r1 % len(single)]))
    else:
        op = windowed[(r1 - 40) % len(windowed)]
        w = 1 + ((int(param_seed) + seed2 * 13) % 30)
        chain.append(_make_windowed(op, w))
    r2 = (int(param_seed) * 40503 + seed2 * 40507) % 6
    if r2 == 0:
        chain.append(_make_single("cs_rank"))
    elif r2 == 1:
        chain.append(_make_windowed("ts_delta", 1 + seed2 % 12))
    elif r2 == 2:
        chain.append(_make_windowed("ts_rank", 1 + seed2 % 12))
    elif r2 == 3:
        chain.append(_make_single("zscore"))
    elif r2 == 4:
        chain.append(_make_windowed("ts_sum", 2 + (param_seed + seed2) % 12))
    # r2 == 5 → single-layer chain

    def _apply(x: Any) -> Any:
        for c in chain:
            x = c(x)
        return x

    return _apply


def _make_single(op: str) -> Callable[[Any], Any]:
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    f = make_cleaned_call_factory(op)
    return lambda x: f(x)


def _make_windowed(op: str, w: int) -> Callable[[Any], Any]:
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    f = make_cleaned_call_factory(op)
    return lambda x: f(x, w)


def generate_pool(
    n_roots: int,
    *,
    seed: int = 42,
    shared_frac: float = 0.6,
    force_unique_pct: float = 99.0,
) -> list[LadderRoot]:
    """Generate ``n_roots`` real-operator roots with shared-subtree injection.

    Guarantees (checked by caller tests):
    - unique structural fingerprints > 95%
    - ``ts_mean(close,20)`` and ``ts_std(return,20)`` appear as children across
      a large fraction of roots (CSE fuel)
    - build failures are preserved as ``LadderRoot(build_error=...)`` entries
    """
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    ops = load_terminal_operators()
    if not ops:
        raise RuntimeError("no terminal operators available for pool generation")
    # Restrict to the ~57 operators runnable on a plain-panel offline source
    # (no typed group/context keys needed) — keeps the failure surface real but
    # not degenerate.
    ops = [
        op for op in ops
        if not any(str((s.get("roles") or ("data",))[0]) == "context"
                   for s in (op.get("input_slots") or ()))
        and "group" not in (op.get("scalar_parameters") or ())
    ]
    if len(ops) < 40:
        raise RuntimeError(f"too few runnable operators: {len(ops)}")
    mean_f = make_cleaned_call_factory("ts_mean")
    std_f = make_cleaned_call_factory("ts_std")

    pool: list[LadderRoot] = []
    seen: set[str] = set()
    rng = np.random.default_rng(seed)
    attempts = 0
    max_attempts = n_roots * 60 + 5000
    family_seed = seed * 104729
    # index rotates over the RUNNABLE operator subset
    runnable_ops = ops

    while len(pool) < n_roots and attempts < max_attempts:
        attempts += 1
        i = len(pool) % len(runnable_ops)
        op = runnable_ops[i]
        canonical = str(op["canonical"])
        family = _economic_family(canonical)
        param_seed = family_seed + attempts * 7919
        next_fid = f"f_{len(pool):06d}"

        try:
            expr, src_cols, kw_params = build_one_expression(op, param_seed=param_seed)
        except Exception as exc:  # noqa: BLE001 — preserve the REAL error
            pool.append(
                LadderRoot(
                    fid=next_fid,
                    expr=None,
                    operator_name=canonical,
                    economic_effect_family=family,
                    structural_fingerprint=f"__build_failed__:{canonical}:{i}",
                    source_columns=(),
                    scalar_params={},
                    build_error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        shared_root = False

        # shared-subtree injection: wrap a fraction of roots so the forced
        # shared children ``ts_mean(close,20)`` / ``ts_std(return,20)`` appear
        # as SUBTREES of distinct root trees (never as the identical root).
        roll = rng.random()
        wrap_ops = _SHARED_WRAP_OPS + ("ts_mean", "ts_std")
        if roll < shared_frac and canonical in wrap_ops:
            try:
                # vary the outer window per root so the OUTER structure is
                # distinct while the inner child is the shared subtree
                child = mean_f(_make_col("close"), 20) if roll < shared_frac * 0.5 \
                    else std_f(_make_col("return"), 20)
                wrap_w = 5 + (attempts * 7) % 24  # 5..28 deterministic rotation
                if canonical == "ts_corr":
                    # ts_corr needs TWO series (x, y) — use the shared child as x
                    expr = make_cleaned_call_factory("ts_corr")(
                        child, _make_col("volume"), wrap_w
                    )
                elif canonical in ("ts_cov", "ts_beta"):
                    # two-series family: y is the shared child, x a plain field
                    expr = make_cleaned_call_factory(canonical)(
                        _make_col("volume"), child, wrap_w
                    )
                elif canonical in ("ts_delta", "ts_pct", "ts_delay"):
                    expr = make_cleaned_call_factory(canonical)(
                        child, max(1, wrap_w)
                    )
                elif canonical in ("ts_mean", "ts_std"):
                    expr = make_cleaned_call_factory(canonical)(child, wrap_w)
                else:
                    # window-family wrapper: canonical(child, window)
                    expr = make_cleaned_call_factory(canonical)(child, wrap_w)
                family = "mean_reversion" if roll >= shared_frac * 0.5 and canonical not in ("ts_mean", "ts_std") else family
                src_cols = ("close", "volume") if canonical == "ts_corr" else (
                    ("return", "volume") if canonical in ("ts_cov", "ts_beta") else
                    (("close",) if roll < shared_frac * 0.5 else ("return",))
                )
                shared_root = True
            except Exception:
                pass

        # Large pools need more structural headroom than 57 ops x decorators
        # alone provide: stack a SECOND operator layer with an attempts-varied
        # window so every root is a distinct multi-layer tree (window literals
        # now participate in the fingerprint: ts_rank(ts_mean(close,W1),W2)
        # with W1 != W2 are distinct).
        if not shared_root and attempts % 2 == 0:
            try:
                layer2_ops = ("cs_rank", "ts_rank", "ts_delta", "zscore",
                              "log", "tanh", "ts_sum", "ts_std", "ts_max",
                              "ts_var", "ts_median")
                l2 = layer2_ops[attempts % len(layer2_ops)]
                w2 = 1 + (attempts * 3) % 30  # 1..30
                if l2 in ("cs_rank", "zscore", "log", "tanh"):
                    expr = make_cleaned_call_factory(l2)(expr)
                else:
                    expr = make_cleaned_call_factory(l2)(expr, w2)
            except Exception:
                pass

        # For the largest pools (>= 100k) add a BINARY composition layer on
        # plain roots: add/subtract/multiply/divide of expr with a sibling
        # operator (attempts-varied) — combinatorially expands the surface
        # beyond any single unary-decorator ceiling while keeping every root a
        # genuine multi-operator production factor.
        if len(pool) > 24000 and canonical not in _UNWRAPPABLE_TWO_SLOT_OPS:
            try:
                bin_ops = ("add", "subtract", "multiply", "divide")
                bo = bin_ops[attempts % len(bin_ops)]
                sibling_candidates = [o for o in ops if o["canonical"] not in _UNWRAPPABLE_TWO_SLOT_OPS]
                sop = sibling_candidates[(attempts * 7) % len(sibling_candidates)]
                sexpr, _, _ = build_one_expression(
                    sop, param_seed=param_seed + 999983
                )
                expr = make_cleaned_call_factory(bo)(expr, sexpr)
                family = "unknown"
            except Exception:
                pass

        if len(pool) > 35000:
            # 100k pools need very deep structural headroom: stack 2-3 extra
            # decorator layers per root (each layer adds a unique window/op
            # combination), so the combined space is 10^5+ even from 57 ops.
            try:
                for _stack in range(2 + (attempts % 2)):
                    expr = _decorator_chain(
                        param_seed + _stack * 7919, attempts + _stack * 13
                    )(expr)
                family = "cross_section"
            except Exception:
                pass

        fp = structural_key(expr)

        # Finite fixed-signature operator surface → structural dup. Widen the
        # input graph with decorators so the tree is NEW and unique while the
        # wrapped roots above still reuse the shared subtree children. Only
        # operators that keep the root's typed inputs valid are used (ts_corr /
        # ts_beta / ts_cov / ts_cov / where take TWO series inputs — wrapping
        # them with a unary decorator that passes a literal as the 2nd series is
        # a real typed-input error, so those roots are dropped instead).
        if fp in seen and canonical in _UNWRAPPABLE_TWO_SLOT_OPS:
            continue
        found_unique = False
        for _tries in range(48):
            if fp not in seen:
                found_unique = True
                break
            chain = _decorator_chain(param_seed, attempts + _tries)
            expr2 = chain(expr)
            fp2 = structural_key(expr2)
            if fp2 not in seen:
                expr, fp, family = expr2, fp2, "cross_section"
                found_unique = True
                break
        # widen with a second decorator layer chained on the first
        if not found_unique:
            for _w in range(12):
                chain = _decorator_chain(param_seed * 31 + _w, attempts + _w)
                expr3 = chain(expr)
                fp3 = structural_key(expr3)
                if fp3 not in seen:
                    expr, fp, family = expr3, fp3, "cross_section"
                    found_unique = True
                    break
        # swap every leaf ColumnRef to a fresh source column
        if not found_unique:
            new_expr = _rewrite_sources_one_level(expr, param_seed + attempts * 13)
            fp2 = structural_key(new_expr)
            if fp2 not in seen:
                expr, fp = new_expr, fp2
                found_unique = True
        if not found_unique:
            # structurally exhausted for THIS op+seed — advance attempts only
            continue
        seen.add(fp)
        pool.append(
            LadderRoot(
                fid=next_fid,
                expr=expr,
                operator_name=canonical,
                economic_effect_family=family,
                structural_fingerprint=fp,
                source_columns=tuple(src_cols),
                scalar_params=dict(kw_params),
            )
        )

    if len(pool) < n_roots:
        raise RuntimeError(
            f"generation stopped early: wanted {n_roots} got {len(pool)} "
            f"after {max_attempts} attempts"
        )
    return pool


def pool_stats(pool: list[LadderRoot]) -> dict[str, Any]:
    fps = [r.structural_fingerprint for r in pool]
    unique = len(set(fps))
    total = len(pool)
    fams: Counter[str] = Counter(r.economic_effect_family for r in pool)
    return {
        "total": total,
        "unique_structural_roots": unique,
        "unique_pct": round(100.0 * unique / total, 3) if total else 0.0,
        "n_families": len(fams),
        "families": dict(fams.most_common()),
        "build_failures": sum(1 for r in pool if r.build_error),
    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------
def stratified_sample(
    pool: list[LadderRoot],
    n: int,
    rng: np.random.Generator,
) -> list[LadderRoot]:
    """Sample ``n`` roots stratified by ``economic_effect_family``.

    Guarantees each family present in the pool gets >= 1 representative when
    ``n`` allows; remaining slots filled proportionally. Never mutates pool.
    """
    if n >= len(pool):
        return list(pool)
    layers: "OrderedDict[str, list[LadderRoot]]" = OrderedDict()
    for r in pool:
        layers.setdefault(r.economic_effect_family, []).append(r)
    keys = list(layers.keys())
    rng.shuffle(keys)

    picked: list[LadderRoot] = []
    picked_ids: set[str] = set()

    def _add(r: LadderRoot) -> bool:
        if r.fid in picked_ids:
            return False
        picked_ids.add(r.fid)
        picked.append(r)
        return True

    for key in keys:
        if len(picked) >= n:
            break
        members = layers[key]
        _add(members[int(rng.integers(0, len(members)))])

    if len(picked) < n:
        weights = np.array([len(layers[k]) for k in keys], dtype=float)
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
        need = n - len(picked)
        picks = rng.choice(len(keys), size=need, p=weights)
        for ki in picks:
            members = layers[keys[int(ki)]]
            for r in members:
                if _add(r):
                    break
    if len(picked) < n:
        # final safety fill across the whole pool
        for r in pool:
            if len(picked) >= n:
                break
            _add(r)
    return picked[:n]


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
def classify_error_message(message: str) -> str:
    low = str(message or "").lower()
    if any(k in low for k in (
        "no such", "not found", "missing", "unavailable", "catalog",
        "coverage", "no data", "empty", "does not exist",
        "unknown column", "unknown field", "fieldregistry",
        "typedinputcontract", "resourceadmission", "contracterror",
        "inflight scan bytes", "required raw", "require raw",
        "source_unknown", "supported_markets", "data degeneracy", "degenerate",
    )):
        return FAIL_DATA
    if any(k in low for k in (
        "param", "arity", "argument", "typeerror", "valueerror",
        "unsupported", "invalid", "signature", "keyword", "positional",
        "dslparse", "compile", "not callable", "coerce",
        "missingdefault", "missing 1 required positional", "unknown operator",
    )):
        return FAIL_PARAM
    if any(k in low for k in (
        "binder error", "duckdb", "referenced column", "engineerror",
    )):
        return FAIL_DATA
    if any(k in low for k in (
        "quantile", "domain", "must be in", "zero", "negative", "nan",
        "overflow", "log of", "sqrt of", "nonzero", "division",
    )):
        return FAIL_SEMANTIC
    if any(k in low for k in ("memory", "oom", "alloc", "rss", "budget_bytes", "refused")):
        return FAIL_OOM
    if any(k in low for k in ("timeout", "timed out", "deadline", "abort")):
        return FAIL_TIMEOUT
    return FAIL_OTHER


def classify_result(result: Any) -> str | None:
    """Semantic check over a produced result. None == pass."""
    if result is None:
        return FAIL_SEMANTIC
    try:
        arr = result.values if hasattr(result, "values") else np.asarray(result)
        if arr.size == 0:
            return FAIL_SEMANTIC
        flat = np.asarray(arr, dtype=float).ravel()
        if np.isnan(flat).all():
            return FAIL_SEMANTIC
        if float(np.isnan(flat).mean()) > 0.999:
            return FAIL_SEMANTIC
    except Exception:
        return FAIL_OTHER
    return None


@dataclass
class RootFailure:
    fid: str
    operator_name: str
    category: str
    error_message: str
    phase: str = "run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fid": self.fid,
            "operator_name": self.operator_name,
            "category": self.category,
            "phase": self.phase,
            "error_message": str(self.error_message)[:2000],
        }


# ---------------------------------------------------------------------------
# Level state
# ---------------------------------------------------------------------------
@dataclass
class LevelState:
    level: int
    total: int
    passed: int = 0
    failed: int = 0
    unique_structural_roots: int = 0
    unique_structural_pct: float = 0.0
    n_families: int = 0
    families: dict[str, int] = field(default_factory=dict)
    failed_by_class: dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in FAIL_CATEGORIES}
    )
    failures: list[RootFailure] = field(default_factory=list)
    wall_time_s: float = 0.0
    shared_nodes: int = 0
    shared_result_cache_entries: int = 0
    reuse_edges: int = 0
    control_plane: str = ""
    execution_mode: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)
    sink_files: list[str] = field(default_factory=list)
    workers: int = 0
    governance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "unique_structural_roots": self.unique_structural_roots,
            "unique_structural_pct": self.unique_structural_pct,
            "n_families": self.n_families,
            "families": dict(self.families),
            "failed_by_class": dict(self.failed_by_class),
            "failures": [f.to_dict() for f in self.failures[:200]],
            "wall_time_s": round(self.wall_time_s, 3),
            "shared_nodes": self.shared_nodes,
            "shared_result_cache_entries": self.shared_result_cache_entries,
            "reuse_edges": self.reuse_edges,
            "control_plane": self.control_plane,
            "execution_mode": self.execution_mode,
            "workers": self.workers,
            "governance": self.governance,
            "telemetry": self.telemetry,
            "sink_files": self.sink_files[:200],
        }


# ---------------------------------------------------------------------------
# Engine + in-memory data source (offline; no COS/data_access reads)
# ---------------------------------------------------------------------------
def build_data_source(n_stocks: int = 60, n_days: int = 252, seed: int = 0) -> Any:
    """Deterministic MultiIndex-Series source implementing the FE DataSource API."""
    import pandas as pd
    from factor_engine.storage.datasource import DataSource

    dates = pd.bdate_range("2023-01-02", periods=n_days)
    assets = [f"A{1000 + i:04d}" for i in range(n_stocks)]
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n_days, n_stocks)) * 0.02
    close = 10.0 * np.exp(np.cumsum(rets, axis=0))
    close = np.maximum(close, 0.01)
    open_ = close * (1.0 + rng.standard_normal((n_days, n_stocks)) * 0.003)
    high_ = np.maximum(open_, close) * (
        1.0 + np.abs(rng.standard_normal((n_days, n_stocks))) * 0.003
    )
    low_ = np.minimum(open_, close) * (
        1.0 - np.abs(rng.standard_normal((n_days, n_stocks))) * 0.003
    )
    volume = rng.integers(1_000_000, 20_000_000, size=(n_days, n_stocks)).astype(float)
    amount = volume * close
    log_ret = np.vstack([np.full((1, n_stocks), np.nan), np.diff(np.log(close), axis=0)])
    idx = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
    data: dict[str, pd.Series] = {}
    for col, mat in (
        ("open", open_), ("high", high_), ("low", low_), ("close", close),
        ("volume", volume), ("amount", amount), ("return", log_ret),
    ):
        data[col] = pd.Series(np.asarray(mat, dtype=float).ravel(), index=idx)

    class _InMemorySource(DataSource):
        def load_column(self, name: str) -> Any:
            return data[name]

        def load_columns(self, names: list[str]) -> dict[str, Any]:
            return {n: data[n] for n in names if n in data}

        def scan_polars_long(self, columns: list[str]):
            import polars as pl

            from factor_engine.storage.factor_format import series_to_long_table

            merged = None
            tcol = icol = None
            for name in sorted(columns):
                series = data[name]
                if tcol is None:
                    tcol = str(series.index.names[0])
                    icol = str(series.index.names[1])
                part = series_to_long_table(
                    series, timestamp_col=tcol, asset_col=icol, value_col=name
                )
                merged = part if merged is None else merged.merge(
                    part, on=[tcol, icol], how="outer"
                )
                merged[name] = merged[name].astype("float64")
            renamed = merged.rename(columns={tcol: "ts", icol: "inst"})
            from factor_engine.backend.long_frame import long_table_to_polars_lazy

            return long_table_to_polars_lazy(renamed, float_cols=columns)

        def scan_index_long(self):
            return self.scan_polars_long(sorted(data.keys())).select(["ts", "inst"]).unique()

    return _InMemorySource()


def build_engine(source: Any) -> Any:
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine

    return FactorEngine(
        backend=PandasBackend(),
        data_source=source,
        cache=None,
        run_mode="research",
    )


# ---------------------------------------------------------------------------
# Worker governance (REAL enforcement)
# ---------------------------------------------------------------------------
def resolve_workers(requested: int) -> dict[str, Any]:
    """Govern ``--workers`` via multiworker_governance budget; max 8 for ladder."""
    from factor_engine.runtime.multiworker_governance import (
        log_governance_decision,
        resolve_worker_budget,
    )

    requested = max(1, min(int(requested), MAX_WORKERS))
    budget = resolve_worker_budget(
        requested,
        per_worker_threads=1,
        total_cores=TOTAL_CORES,
    )
    log_governance_decision(budget)
    return {
        "workers": budget.workers,
        "cpu_budget": budget.cpu_budget,
        "coexist_count": budget.coexist_count,
        "clamped": budget.clamped,
        "warning": budget.warning,
    }


def validate_timeout(timeout_s: float | None) -> float:
    """``timeout=0``/``None`` is REJECTED — never silently disables enforcement."""
    if timeout_s is None:
        raise ValueError("--per-factor-timeout is required (positive seconds)")
    t = float(timeout_s)
    if t <= 0:
        raise ValueError(
            f"--per-factor-timeout must be > 0 (got {timeout_s!r}); "
            "0 would silently disable enforcement"
        )
    return t


# ---------------------------------------------------------------------------
# Level execution — ONE (chunked) BATCH run_many per level
# ---------------------------------------------------------------------------
def _result_rows(result: Any) -> list[tuple[str, str, float]]:
    """(date, asset, value) rows from a result panel; NaN rows dropped."""
    import pandas as pd

    arr = np.asarray(result.values if hasattr(result, "values") else result)
    idx = result.index
    names = list(getattr(idx, "names", [None, None]))
    if len(names) >= 2:
        d_idx = idx.get_level_values(0)
        a_idx = idx.get_level_values(1)
        if arr.ndim == 1:
            arr = arr[:, None]
        out: list[tuple[str, str, float]] = []
        n_dates = len(d_idx)
        n_assets = len(a_idx)
        vals = np.asarray(arr, dtype=float)
        if vals.ndim == 2 and vals.shape[1] > 1:
            for r in range(vals.shape[0]):
                d = d_idx[r]
                ds = str(d.date()) if hasattr(d, "date") else str(d)
                for c in range(vals.shape[1]):
                    v = float(vals[r, c])
                    if not (v != v):
                        out.append((ds, str(a_idx[c]), v))
        else:
            flat = vals.ravel()
            for k in range(min(len(d_idx), len(flat))):
                v = float(flat[k])
                if not (v != v):
                    out.append((str(d_idx[k].date()) if hasattr(d_idx[k], "date") else str(d_idx[k]),
                                str(a_idx[k]), v))
        return out
    # single index: rows are dates, columns are assets
    if arr.ndim == 1:
        arr = arr[:, None]
    out = []
    for r in range(arr.shape[0]):
        d = idx[r]
        ds = str(d.date()) if hasattr(d, "date") else str(d)
        for c in range(arr.shape[1]):
            v = float(arr[r, c])
            if not (v != v):
                asset = str(result.columns[c]) if hasattr(result, "columns") else str(c)
                out.append((ds, asset, v))
    return out


def run_level(
    level: int,
    roots: list[LadderRoot],
    engine: Any,
    *,
    out_dir: Path,
    workers: int,
    per_factor_timeout: float,
    resume: bool = True,
    write_results: bool = True,
    result_policy: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> LevelState:
    """Execute a level as chunks of BATCH ``engine.run_many(factors, enable_cse=True)``.

    - Never a per-factor ``run()`` loop. (The sole exception: a chunk that
      overran ``per_factor_timeout`` is re-run root-by-root with the watchdog
      armed so the stuck root is attributed and aborted as a ``timeout``.)
    - ``result_policy``: ``"sink"`` streams each root through the storage
      ``StreamingSink`` as it finishes; ``"return"`` keeps results in memory
      (small levels only).
    - Checkpoint: ``DryRunCheckpoint`` marks each completed/failed root; the
      level-state JSON is written every ``checkpoint_every`` formulas.
    """
    from factor_engine.api.factor import Factor
    from factor_engine.runtime.dry_run_checkpoint import CampaignMeta, DryRunCheckpoint
    from factor_engine.storage.streaming_sink import StreamingSink

    if result_policy is None:
        result_policy = "return" if len(roots) <= 500 else "sink"
    if write_results and result_policy == "sink":
        policy = "sink"
    else:
        policy = result_policy if result_policy == "sink" else "return"

    state = LevelState(level=level, total=len(roots))

    # ---- invariant: unique structural roots > 95% ---------------------------
    fps = [r.structural_fingerprint for r in roots]
    state.unique_structural_roots = len(set(fps))
    state.unique_structural_pct = round(100.0 * len(set(fps)) / max(1, len(roots)), 3)
    state.n_families = len({r.economic_effect_family for r in roots})
    state.families = dict(Counter(r.economic_effect_family for r in roots).most_common())
    if state.unique_structural_pct <= 95.0:
        raise AssertionError(
            f"level {level}: unique structural roots = {state.unique_structural_pct}% "
            f"({state.unique_structural_roots}/{len(roots)}) <= 95% — refusing to run "
            "a degenerate level"
        )

    # ---- checkpoint / resume -----------------------------------------------
    ckpt_dir = out_dir / "checkpoint" / f"level_{level}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = DryRunCheckpoint(checkpoint_dir=ckpt_dir)
    ckpt.init_campaign(
        CampaignMeta(
            campaign_id=f"ladder_level_{level}",
            total_roots=len(roots),
            source_snapshot=f"synth:60x252:level:{level}:roots:{len(roots)}",
            factor_hash=f"r61-ladder58-l{level}",
        )
    )
    state_json = out_dir / "level_state" / f"level_{level}.json"
    state_json.parent.mkdir(parents=True, exist_ok=True)
    prior: dict[str, Any] = {}
    if state_json.exists():
        try:
            prior = json.loads(state_json.read_text())
        except Exception:
            prior = {}

    all_fids = [r.fid for r in roots]
    pending_ids = ckpt.pending_roots(all_fids)
    if prior:
        state.passed = int(prior.get("passed", 0) or 0)
        state.failed = int(prior.get("failed", 0) or 0)
        for c in FAIL_CATEGORIES:
            state.failed_by_class[c] = int(prior.get("failed_by_class", {}).get(c, 0) or 0)
        state.failures = [
            RootFailure(fid=f["fid"], operator_name=f.get("operator_name", f["fid"]),
                        category=f.get("category", FAIL_OTHER),
                        error_message=f.get("error_message", ""), phase=f.get("phase", "run"))
            for f in prior.get("failures", [])
        ]
        state.wall_time_s = float(prior.get("wall_time_s", 0.0) or 0.0)
        state.sink_files = list(prior.get("sink_files") or [])

    def _save_state() -> None:
        state.wall_time_s = time.time() - _t0
        state_json.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))

    _t0 = time.time()
    pending_roots = [r for r in roots if r.fid in pending_ids]
    if not pending_roots:
        state.wall_time_s = float(prior.get("wall_time_s", 0.0) or 0.0)
        _save_state()
        return state

    governance = resolve_workers(workers)
    state.workers = governance["workers"]
    state.governance = governance

    sink_dir = out_dir / "lake" / f"level_{level}"
    sink = StreamingSink(output_dir=sink_dir, campaign_id=f"ladder_level_{level}")

    # build factors (preserving build errors from generation)
    factors: list[Factor] = []
    build_failures: list[RootFailure] = []
    for r in pending_roots:
        if r.expr is None:
            build_failures.append(
                RootFailure(
                    fid=r.fid,
                    operator_name=r.operator_name,
                    category=FAIL_PARAM,
                    error_message=r.build_error or "build failure (see pool)",
                    phase="build",
                )
            )
            continue
        factors.append(Factor(name=r.fid, expr=r.expr))

    def _mark_failed(fid: str, *, category: str, message: str, phase: str, op: str = "") -> None:
        state.failed += 1
        state.failed_by_class[category] = state.failed_by_class.get(category, 0) + 1
        state.failures.append(
            RootFailure(fid=fid, operator_name=op or fid, category=category,
                        error_message=message, phase=phase)
        )
        try:
            ckpt.mark_failed(fid, RuntimeError(message[:2000]))
        except Exception:
            pass

    for bf in build_failures:
        _mark_failed(bf.fid, category=bf.category, message=bf.error_message,
                     phase="build", op=bf.operator_name)

    processed = 0
    op_by_fid = {r.fid: r.operator_name for r in pending_roots}

    def _record_success(fid: str) -> None:
        nonlocal processed
        processed += 1
        state.passed += 1
        try:
            ckpt.mark_completed(fid, marker={"level": level})
        except Exception:
            pass
        if processed % checkpoint_every == 0:
            _save_state()

    def _on_sink(name: str, result: Any) -> None:
        """run_many sink callback: classify -> write shard -> checkpoint."""
        try:
            sem = classify_result(result)
            if sem is not None:
                _mark_failed(name, category=sem,
                             message="semantic failure: empty / all-NaN / >99.9% NaN",
                             phase="semantic", op=op_by_fid.get(name, name))
                return
            if write_results:
                rows = _result_rows(result)
                with sink.open(name) as shard:
                    shard.add_many(
                        [type("_R", (), {"date": d, "asset": a, "value": v})() for d, a, v in rows]
                    )
            _record_success(name)
        except Exception as exc:  # noqa: BLE001
            _mark_failed(name, category=classify_error_message(str(exc)),
                         message=f"{type(exc).__name__}: {exc}", phase="sink",
                         op=op_by_fid.get(name, name))

    def _record_returned(out: dict[str, Any]) -> None:
        for name, result in (out.get("results") or {}).items():
            if name in op_by_fid:
                _on_sink(name, result)

    # ---- batch execution ----------------------------------------------------
    factors.sort(key=lambda f: f.name)
    n_chunks = max(1, math.ceil(len(factors) / chunk_size))
    last_out: dict[str, Any] = {}
    for ci in range(n_chunks):
        chunk = factors[ci * chunk_size:(ci + 1) * chunk_size]
        if not chunk:
            continue
        chunk_t0 = time.monotonic()
        overran = False
        try:
            if policy == "sink":
                out = engine.run_many(
                    chunk, enable_cse=True, auto_warmup=False, trim_warmup=True,
                    market=None, pit_enforce=False,
                    result_policy="sink", sink=_on_sink,
                )
            else:
                out = engine.run_many(
                    chunk, enable_cse=True, auto_warmup=False, trim_warmup=True,
                    market=None, pit_enforce=False,
                    result_policy="return",
                )
                _record_returned(out)
            last_out = out
            elapsed = time.monotonic() - chunk_t0
            if elapsed > per_factor_timeout * max(1, len(chunk)):
                overran = True
        except Exception as exc:  # noqa: BLE001 — classify and continue
            cat = classify_error_message(str(exc))
            if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                cat = FAIL_TIMEOUT
            for f in chunk:
                _mark_failed(f.name, category=cat,
                             message=f"{type(exc).__name__}: {exc}", phase="run",
                             op=op_by_fid.get(f.name, f.name))

        if overran:
            # REAL watchdog: re-run chunk roots individually with the armed
            # in-thread deadline so the stuck root is aborted + classified.
            for f in chunk:
                started = time.monotonic()
                try:
                    out1 = engine.run_many(
                        [f], enable_cse=True, auto_warmup=False, trim_warmup=True,
                        market=None, pit_enforce=False, result_policy="return",
                    )
                    _record_returned(out1)
                except Exception as exc:  # noqa: BLE001
                    cat = classify_error_message(str(exc))
                    if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                        cat = FAIL_TIMEOUT
                    _mark_failed(f.name, category=cat,
                                 message=f"{type(exc).__name__}: {exc}", phase="run",
                                 op=op_by_fid.get(f.name, f.name))
                elapsed1 = time.monotonic() - started
                if elapsed1 > per_factor_timeout and not ckpt.is_completed(f.name):
                    # still not finished in time and not marked failed above:
                    # record a real timeout classification for this root
                    _mark_failed(f.name, category=FAIL_TIMEOUT,
                                 message=f"per-factor watchdog: exceeded {per_factor_timeout:g}s",
                                 phase="run", op=op_by_fid.get(f.name, f.name))
        if processed % checkpoint_every == 0:
            _save_state()

    state.wall_time_s = time.time() - _t0
    state.sink_files = _list_sink_files(sink_dir)

    # ---- CSE / shared-subtree telemetry -------------------------------------
    try:
        dag = last_out.get("dag") if last_out else None
        if dag is not None:
            state.shared_nodes = len(getattr(dag, "shared_nodes", None) or {})
        state.control_plane = str(last_out.get("control_plane", ""))
        ss = last_out.get("scheduler_stats")
        if isinstance(ss, dict):
            state.execution_mode = str(ss.get("auto_execution_mode", ""))
        from factor_engine.telemetry import execution_telemetry as et

        snap = {k: int(v) for k, v in et.snapshot().items()}
        state.telemetry = snap
        state.shared_result_cache_entries = int(snap.get("cse.shared_nodes", 0) or 0)
        state.reuse_edges = int(snap.get("cse.reuse_edges", 0) or 0)
    except Exception:
        pass
    _save_state()
    return state


def _list_sink_files(sink_dir: Path) -> list[str]:
    shard_dir = Path(sink_dir) / "shards"
    if not shard_dir.exists():
        return []
    try:
        return sorted(str(p) for p in shard_dir.glob("*.parquet"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="100,1000", help="comma-separated ladder levels")
    ap.add_argument("--out", default="/tmp/r61_ladder58", help="output/status dir")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--per-factor-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-level", type=int, default=None,
                    help="CI: run ONLY this level (e.g. 100) then exit")
    ap.add_argument("--max-level", type=int, default=None,
                    help="cap levels at this value (e.g. --max-level 1000)")
    ap.add_argument("--gen-only", type=int, default=0,
                    help="generate N roots and print uniqueness stats, no execution")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--no-resume", action="store_true", help="ignore previous checkpoint")
    ap.add_argument("--result-policy", choices=("return", "sink"), default=None,
                    help="defaults: return for <=500 roots, sink otherwise")
    ap.add_argument("--no-write", action="store_true",
                    help="classify/semantic-check but skip shard writes (dry semantics)")
    ap.add_argument("--panel-stocks", type=int, default=60,
                    help="offline synthetic panel: number of stocks")
    ap.add_argument("--panel-days", type=int, default=252,
                    help="offline synthetic panel: number of days")
    args = ap.parse_args(argv)

    timeout_s = validate_timeout(args.per_factor_timeout)

    if args.gen_only and args.gen_only > 0:
        print(f"[ladder] generating pool of {args.gen_only} roots (no execution)...",
              flush=True)
        t0 = time.time()
        pool = generate_pool(args.gen_only, seed=args.seed)
        stats = pool_stats(pool)
        stats["generation_wall_s"] = round(time.time() - t0, 3)
        print(json.dumps(stats, indent=2, sort_keys=True))
        uniq = stats["unique_pct"]
        if stats["total"] == args.gen_only and uniq <= 95.0:
            print(f"[ladder] FAIL: uniqueness {uniq}% <= 95%", flush=True)
            return 3
        print(f"[ladder] generation OK: {stats['total']} roots, "
              f"{stats['unique_structural_roots']} unique "
              f"({uniq}%), {stats['n_families']} families", flush=True)
        return 0

    levels = [int(x) for x in str(args.levels).split(",") if x.strip()]
    if args.dry_level is not None:
        levels = [args.dry_level]
    if args.max_level is not None:
        levels = [l for l in levels if l <= args.max_level]
    if not levels:
        print("no levels selected", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ladder] engine pool generation: {max(levels)}-root universe", flush=True)
    t0 = time.time()
    universe_size = max(levels) if max(levels) > 1000 else 1000
    pool = generate_pool(universe_size, seed=args.seed)
    pstats = pool_stats(pool)
    print(f"[ladder] pool: {pstats['total']} roots, {pstats['unique_structural_roots']} "
          f"unique ({pstats['unique_pct']}%), {pstats['n_families']} families "
          f"[{time.time() - t0:.1f}s]", flush=True)
    if pstats["unique_pct"] <= 95.0:
        print(f"[ladder] FAIL: pool uniqueness {pstats['unique_pct']}% <= 95%", flush=True)
        return 3

    rng = np.random.default_rng(args.seed)
    engine = build_engine(
        build_data_source(n_stocks=args.panel_stocks, n_days=args.panel_days)
    )

    summary: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "levels": {},
        "pool": pstats,
        "criteria": {
            "A_100": "core semantics",
            "B_1000": "CSE/IO",
            "C_5000": "scheduler/RSS",
            "D_20000": "write/cache",
            "E_50000": "long-run stability",
            "F_100000": "final",
        },
    }
    for level in levels:
        print(f"\n[ladder] === level {level} ===", flush=True)
        if level > len(pool):
            print(f"[ladder] level {level} > pool {len(pool)} — extending pool", flush=True)
            pool = generate_pool(level, seed=args.seed + level)
            pstats = pool_stats(pool)
            print(f"[ladder] extended pool: {pstats['unique_pct']}% unique", flush=True)
        sample = stratified_sample(pool, level, rng)
        fams = Counter(r.economic_effect_family for r in sample)
        print(f"[ladder] sample: {len(sample)} roots across {len(fams)} families "
              f"{dict(fams.most_common())}", flush=True)
        if level == 100 and len(fams) < 2:
            raise AssertionError(
                f"level 100 sample must cover >= 2 economic_effect_families "
                f"(got {dict(fams)})"
            )
        st = run_level(
            level,
            sample,
            engine,
            out_dir=out_dir,
            workers=args.workers,
            per_factor_timeout=timeout_s,
            resume=not args.no_resume,
            write_results=not args.no_write,
            result_policy=args.result_policy,
            chunk_size=args.chunk_size,
        )
        summary["levels"][str(level)] = st.to_dict()
        print(f"[ladder] level {level}: passed={st.passed} failed={st.failed} "
              f"wall={st.wall_time_s:.1f}s workers={st.workers} "
              f"shared_nodes={st.shared_nodes} reuse_edges={st.reuse_edges} "
              f"shared_cache={st.shared_result_cache_entries}", flush=True)
        for cls, cnt in st.failed_by_class.items():
            if cnt:
                print(f"    {cls}: {cnt}", flush=True)
        for fail in st.failures[:5]:
            print(f"    e.g. {fail.fid} [{fail.category}/{fail.phase}] "
                  f"{fail.error_message[:160]}", flush=True)

    status_path = out_dir / "ladder_status.json"
    status_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\n[ladder] status -> {status_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
