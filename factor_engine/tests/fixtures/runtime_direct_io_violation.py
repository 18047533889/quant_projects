"""
Negative test fixture mirroring runtime/shard_executor.py direct I/O violation.

This file deliberately violates FE-DA-P0-002 by performing direct parquet I/O
outside the DataAccess boundary, similar to how runtime modules currently bypass
governed storage.

EXPECTED: check_dataaccess_io_boundary.py MUST detect this as a violation.
"""
import pandas as pd
from pathlib import Path


def load_shard_checkpoint_direct(checkpoint_path: Path) -> pd.DataFrame:
    """
    VIOLATION: Direct parquet read bypassing DataAccess governance.

    Mirrors runtime/shard_executor.py:68 pattern where runtime modules
    directly read parquet files instead of using governed DataAccess APIs.

    Missing:
    - PIT validation
    - Schema version checks
    - Access control
    - Audit logging
    - Snapshot consistency
    """
    # VIOLATION: Direct pd.read_parquet bypasses DataAccess
    return pd.read_parquet(checkpoint_path)


def save_intermediate_result_direct(df: pd.DataFrame, output_path: Path) -> None:
    """
    VIOLATION: Direct parquet write bypassing DataAccess governance.

    Similar to runtime/spill_store.py pattern where intermediate results
    are written directly without going through governed storage layer.
    """
    # VIOLATION: Direct to_parquet bypasses atomic commit protocol
    df.to_parquet(output_path, engine="pyarrow")
