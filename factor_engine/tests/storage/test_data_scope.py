# -*- coding: utf-8 -*-
"""compute_data_scope 含 params / snapshot 键。"""

from __future__ import annotations

from storage.data_scope import compute_data_scope


class _FakeSource:
    dataset = "factor_lake"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    params = {"factor_id": "mom_3d"}
    data_snapshot_id = "snap_abc123"


def test_compute_data_scope_includes_params_and_snapshot():
    scope_a = compute_data_scope(_FakeSource())
    src_b = _FakeSource()
    src_b.params = {"factor_id": "other"}
    scope_b = compute_data_scope(src_b)
    assert scope_a != scope_b

    src_c = _FakeSource()
    src_c.data_snapshot_id = "snap_other"
    scope_c = compute_data_scope(src_c)
    assert scope_a != scope_c
