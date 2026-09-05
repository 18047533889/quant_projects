# -*- coding: utf-8 -*-
"""DEPRECATED — superseded by ``scripts/production_factor_ladder.py`` (R61-P0 #58).

The benchmarks ladder used SYNTHETIC pandas ``rolling_mean``/``pct_change(5)``
functions (NOT real FE operators) through the real sink/checkpoint. It is
retained only as a thin compat shim for any remaining imports / historical CI
references. All real ladder logic lives in the production ladder module.
"""

from __future__ import annotations

import warnings
from typing import Any

warnings.warn(
    "benchmarks/dry_run_ladder.py is deprecated (R61-P0 #58): use "
    "factor_engine.scripts.production_factor_ladder instead.",
    DeprecationWarning,
    stacklevel=2,
)

from factor_engine.scripts.production_factor_ladder import (  # noqa: E402,F401
    LadderRoot,
    RootFailure,
    generate_pool,
    load_terminal_operators,
    pool_stats,
    stratified_sample,
    structural_key,
)
from factor_engine.runtime.dry_run_checkpoint import DryRunCheckpoint, CampaignMeta  # noqa: E402,F401
from factor_engine.storage.streaming_sink import StreamingSink, StreamRow  # noqa: E402,F401

# Historical tunables kept for test back-compat (the new ladder sizes its own
# offline panel; these are no-ops but remain assignable via monkeypatch).
N_STOCKS = 60
N_DAYS = 252
_RNG_SEED = 0
MAX_CORES = 4

# Legacy rung names mapped onto the new ladder's nominal levels.
RUNGS = {"tiny": 100, "small": 1000, "medium": 5000, "large": 20000,
         "xlarge": 50000, "full": 100000}


def run_rung(*, name: str, n_factors: int, root_dir, assert_1e12: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Compat entry — executes a real-operator level through the production ladder."""
    import json
    from pathlib import Path

    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine

    from factor_engine.scripts.production_factor_ladder import (
        build_data_source,
        run_level,
        stratified_sample,
    )

    import numpy as np

    pool = generate_pool(n_factors, seed=42)
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=build_data_source(n_stocks=30, n_days=120),
        run_mode="research",
    )
    out_dir = Path(root_dir)
    st = run_level(
        level=RUNGS.get(name, n_factors),
        roots=stratified_sample(pool, n_factors, np.random.default_rng(0)),
        engine=engine,
        out_dir=out_dir,
        workers=4,
        per_factor_timeout=120.0,
    )
    return {
        "rung": name,
        "actual_roots": n_factors,
        "completed_roots": st.passed + st.failed,
        "passed": st.passed,
        "failed": st.failed,
        "failed_by_class": st.failed_by_class,
        "wall_sec": st.wall_time_s,
        "shared_nodes": st.shared_nodes,
        "reuse_edges": st.reuse_edges,
        "status_json": str(out_dir / "ladder_status.json"),
    }


__all__ = [
    "RUNGS", "run_rung", "LadderRoot", "RootFailure", "generate_pool",
    "load_terminal_operators", "pool_stats", "stratified_sample",
    "structural_key", "DryRunCheckpoint", "CampaignMeta", "StreamingSink",
    "StreamRow",
]
