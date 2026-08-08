# -*- coding: utf-8
"""P0-12 catalog error-hierarchy tests (fail-closed production, lenient research)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from storage.sources.data_access_source import (
    CatalogNotConfigured,
    CatalogUnavailable,
    DataAccessSource,
    UnknownFieldSemanticError,
)


def _mock_ds():
    return MagicMock(time_column="ts", instrument_column="inst", schema={})


def test_production_catalog_api_error_is_fatal():
    class BadStore:
        def resolve_fields(self, names, *, dataset=None):
            from data_access.core.exceptions import EngineError

            raise EngineError("duckdb blew up")

        def get_dataset(self, name):
            return _mock_ds()

    src = DataAccessSource(dataset="x", strict_unknown_fields=True)
    with patch("storage.sources.data_access_source._get_store", return_value=BadStore()):
        with pytest.raises(CatalogUnavailable):
            src._resolve_columns(["foo"])


def test_production_catalog_not_configured_is_fatal():
    class NoConfigStore:
        def resolve_fields(self, names, *, dataset=None):
            raise AssertionError("should not reach")

        def get_dataset(self, name):
            return _mock_ds()

    from data_access.core.exceptions import ValidationError

    src = DataAccessSource(dataset="x", strict_unknown_fields=True)
    with patch(
        "data_access.read.semantic_catalog.get_semantic_catalog",
        side_effect=ValidationError("semantic_fields.yaml 不存在：/nope"),
    ):
        with patch("storage.sources.data_access_source._get_store", return_value=NoConfigStore()):
            with pytest.raises(CatalogNotConfigured):
                src._resolve_columns(["foo"])


def test_production_clean_miss_unknown_field_fails_closed():
    class CleanMissStore:
        def resolve_fields(self, names, *, dataset=None):
            from data_access.core.exceptions import ValidationError

            raise ValidationError(
                "字段 'foo' 未在 SemanticFieldCatalog，也不在任何数据集 schema 中。"
            )

        def get_dataset(self, name):
            return _mock_ds()

    src = DataAccessSource(dataset="x", strict_unknown_fields=True)
    with patch("storage.sources.data_access_source._get_store", return_value=CleanMissStore()):
        with pytest.raises(UnknownFieldSemanticError):
            src._resolve_columns(["foo"])


def test_research_catalog_api_error_falls_back_to_raw():
    class BadStore:
        def resolve_fields(self, names, *, dataset=None):
            from data_access.core.exceptions import EngineError

            raise EngineError("duckdb blew up")

        def get_dataset(self, name):
            return _mock_ds()

    src = DataAccessSource(dataset="x", strict_unknown_fields=False)
    with patch("storage.sources.data_access_source._get_store", return_value=BadStore()):
        physical, _ = src._resolve_columns(["foo"])
    assert physical == ["foo"]


def test_research_clean_miss_allows_raw_passthrough():
    class CleanMissStore:
        def resolve_fields(self, names, *, dataset=None):
            from data_access.core.exceptions import ValidationError

            raise ValidationError(
                "字段 'foo' 未在 SemanticFieldCatalog，也不在任何数据集 schema 中。"
            )

        def get_dataset(self, name):
            return _mock_ds()

    src = DataAccessSource(dataset="x", strict_unknown_fields=False)
    with patch("storage.sources.data_access_source._get_store", return_value=CleanMissStore()):
        physical, _ = src._resolve_columns(["foo"])
    assert physical == ["foo"]
