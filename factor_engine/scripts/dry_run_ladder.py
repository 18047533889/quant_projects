#!/usr/bin/env python3
"""DEPRECATED — superseded by ``scripts/production_factor_ladder.py`` (R61-P0 #58).

This module used to be the per-factor ``dry_run_ladder``. It is retained only as
a thin compat shim so existing imports / CI references keep resolving. All real
ladder logic lives in ``factor_engine.scripts.production_factor_ladder`` (batch
``engine.run_many(factors, enable_cse=True)`` per level, real 62-operator pool,
stratified ``LadderRoot`` sampling, enforced workers/timeout, storage
``StreamingSink`` + ``DryRunCheckpoint`` resume, preserved error classification).

Importing this module emits a ``DeprecationWarning``. Update callers to the new
module — the shim will be removed in a later release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/dry_run_ladder.py is deprecated (R61-P0 #58): use "
    "factor_engine.scripts.production_factor_ladder instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Failure-category constants kept for back-compat with any remaining imports.
FAIL_PARAM = "param_error"
FAIL_DATA = "data_missing"
FAIL_TIMEOUT = "timeout"
FAIL_OOM = "oom"
FAIL_SEMANTIC = "semantic"
FAIL_OTHER = "other"
FAIL_CATEGORIES = [FAIL_PARAM, FAIL_DATA, FAIL_TIMEOUT, FAIL_OOM, FAIL_SEMANTIC, FAIL_OTHER]

# Public API delegated to the new production ladder.
from factor_engine.scripts.production_factor_ladder import (  # noqa: E402,F401
    LadderRoot,
    RootFailure,
    generate_pool,
    load_terminal_operators,
    pool_stats,
    stratified_sample,
    structural_key,
)

from factor_engine.runtime.dry_run_checkpoint import DryRunCheckpoint as Checkpoint  # noqa: E402,F401
from factor_engine.runtime.dry_run_checkpoint import CampaignMeta  # noqa: E402,F401
from factor_engine.storage.streaming_sink import StreamingSink  # noqa: E402,F401

__all__ = [
    "FAIL_PARAM", "FAIL_DATA", "FAIL_TIMEOUT", "FAIL_OOM", "FAIL_SEMANTIC",
    "FAIL_OTHER", "FAIL_CATEGORIES",
    "LadderRoot", "RootFailure", "generate_pool", "load_terminal_operators",
    "pool_stats", "stratified_sample", "structural_key",
    "Checkpoint", "CampaignMeta", "StreamingSink",
]
