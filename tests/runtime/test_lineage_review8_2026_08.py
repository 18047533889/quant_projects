# -*- coding: utf-8 -*-
"""Review-8 lineage fixes: non_null_count on DataFrame results (#483) and
deterministic / secret-redacting config hash (#484, #452)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from runtime.lineage import (
    build_run_lineage,
    hash_data_source_config,
    _canonicalize_config_value,
)


def test_non_null_count_dataframe_result():
    """#483: a panel-native DataFrame result must not crash int(Series)."""
    panel = pd.DataFrame(
        [[1.0, np.nan], [2.0, 3.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=["A", "B"],
    )
    lineage = build_run_lineage(
        factor_id="f1",
        factor_name="f1",
        ast_hash="h",
        operator_catalog_hash="oc",
        result=panel,
    )
    assert lineage.row_count == 2
    assert lineage.non_null_count == 3  # 1 nan cell out of 4


def test_non_null_count_series_and_empty():
    """#483: Series and empty results still work."""
    s = pd.Series([1.0, float("nan"), 2.0])
    lineage = build_run_lineage(
        factor_id="f2", factor_name="f2", ast_hash="h",
        operator_catalog_hash="oc", result=s,
    )
    assert lineage.non_null_count == 2

    empty = pd.Series(dtype=float)
    lineage = build_run_lineage(
        factor_id="f3", factor_name="f3", ast_hash="h",
        operator_catalog_hash="oc", result=empty,
    )
    assert lineage.non_null_count == 0


def test_hash_data_source_config_deterministic_and_key_order_free():
    """#484: same config hashes identically regardless of key order."""
    cfg1 = {"type": "data_access", "dataset": "ashare_stock_daily", "fields": {"close": "Close"}}
    cfg2 = {"fields": {"close": "Close"}, "dataset": "ashare_stock_daily", "type": "data_access"}
    assert hash_data_source_config(cfg1) == hash_data_source_config(cfg2)


def test_hash_config_rejects_nan_inf():
    """#484: NaN / Inf are non-canonical identities and must be rejected."""
    with pytest.raises(ValueError):
        hash_data_source_config({"type": "x", "threshold": float("nan")})
    with pytest.raises(ValueError):
        hash_data_source_config({"type": "x", "threshold": float("inf")})


def test_hash_config_redacts_secrets():
    """#452/#484: secrets never enter the identity — two configs differing only
    in a password must hash identically, and the secret must not appear in any
    canonical serialization."""
    cfg_a = {"type": "s3", "dataset": "d", "access_key": "AKIA-SECRET-A", "secret_key": "super-secret-a"}
    cfg_b = {"type": "s3", "dataset": "d", "access_key": "AKIA-SECRET-B", "secret_key": "super-secret-b"}
    assert hash_data_source_config(cfg_a) == hash_data_source_config(cfg_b)
    canonical = _canonicalize_config_value(cfg_a)
    assert "super-secret-a" not in repr(canonical)
    assert canonical["secret_key"] == "<redacted>"


def test_hash_config_canonicalizes_datetime_and_numpy_scalars():
    """#484: datetimes and numpy scalars serialize canonically."""
    cfg = {
        "type": "x",
        "asof": datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc),
        "ratio": np.float64(1.25),
        "count": np.int64(3),
    }
    h = hash_data_source_config(cfg)
    assert h == hash_data_source_config(
        {"type": "x", "asof": "2026-01-01T10:30:00+00:00", "ratio": 1.25, "count": 3}
    )


def test_hash_config_rejects_unsupported_object():
    """#484: an object whose str() carries a memory address must be rejected,
    not silently hashed."""
    class Opaque:
        def __str__(self):  # pragma: no cover - never reached
            return f"<Opaque at {id(self)}>"

    with pytest.raises(TypeError):
        hash_data_source_config({"type": "x", "opaque": Opaque()})
