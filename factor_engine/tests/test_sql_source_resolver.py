# -*- coding: utf-8
"""resolve_pushdown_source 单元测试。"""
from __future__ import annotations

from backend.sql_pushdown.source_resolver import resolve_pushdown_source
from storage.data_access_source import DataAccessSource
from storage.long_table_source import LongTableDataSource


def test_long_table_resolves():
    inner = DataAccessSource(dataset="test_ds")
    wrapped = LongTableDataSource(inner)
    resolved = resolve_pushdown_source(wrapped)
    assert resolved is wrapped
    assert resolved.dataset == "test_ds"


def test_composite_unwraps_anchor():
    from storage.composite_source import CompositeDataSource

    anchor = DataAccessSource(dataset="anchor_ds")
    other = DataAccessSource(dataset="other_ds")
    comp = CompositeDataSource(
        anchor_source="main",
        anchor_column="close",
        sources={"main": anchor, "aux": other},
    )
    assert resolve_pushdown_source(comp) is anchor
