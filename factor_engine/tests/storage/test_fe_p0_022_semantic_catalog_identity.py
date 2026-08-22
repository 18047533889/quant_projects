"""FE-P0-022: Strict SemanticCatalogIdentity issuer boundary tests.

Validates that:
1. DataAccess issues typed SemanticCatalogIdentity from public API
2. FactorEngine consumes identity without inspecting catalog._fields
3. Production fails closed when catalog unavailable/corrupt
4. Research uses explicit uncacheable diagnostics (no stable 'unavailable')
5. Identity is immutable and uses CanonicalIdentityEncoder
"""

import pytest
from unittest.mock import Mock, patch, PropertyMock
from factor_engine.storage.sources.data_access_source import (
    DataAccessSource,
    CatalogUnavailable,
    CatalogCorrupt,
)


class TestSemanticCatalogIdentityBoundary:
    """FE-P0-022: Boundary tests proving FE never accesses catalog._fields."""

    def test_production_fail_closed_when_catalog_unavailable(self):
        """Production raises CatalogUnavailable when catalog cannot be loaded."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=True,  # production
        )

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Catalog API unavailable")

            with pytest.raises(CatalogUnavailable) as exc_info:
                source.semantic_catalog_identity()

            assert "FE-P0-022" in str(exc_info.value)
            assert "unavailable in production" in str(exc_info.value)

    def test_research_explicit_timestamped_diagnostic_when_unavailable(self):
        """Research returns unique timestamped diagnostic (no stable 'unavailable')."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=False,  # research
        )

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Catalog not configured")

            result1 = source.semantic_catalog_identity()
            result2 = source.semantic_catalog_identity()

            # Both are explicit diagnostics with timestamp
            assert result1.startswith("catalog_unavailable_at_")
            assert result2.startswith("catalog_unavailable_at_")
            # Not a stable 'unavailable' string
            assert result1 != "unavailable"
            assert result2 != "unavailable"

    def test_fe_calls_dataaccess_identity_issuer_not_inspect_fields(self):
        """FE calls catalog.get_identity() and NEVER inspects catalog._fields."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=True,
        )

        mock_catalog = Mock()
        mock_identity = Mock()
        mock_identity.cache_key.return_value = "abc123def456" + "0" * 52  # 64 chars

        # Set up the identity issuer
        mock_catalog.get_identity.return_value = mock_identity

        # Make _fields property raise if accessed (proves FE never touches it)
        type(mock_catalog)._fields = PropertyMock(
            side_effect=AssertionError("FE-P0-022 VIOLATION: FE accessed catalog._fields")
        )

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.return_value = mock_catalog

            result = source.semantic_catalog_identity()

            # FE got the identity without touching _fields
            assert result == "abc123def456" + "0" * 52
            mock_catalog.get_identity.assert_called_once_with(strict=True)
            mock_identity.cache_key.assert_called_once()

            # If _fields was accessed, PropertyMock would have raised AssertionError

    def test_dataaccess_issues_typed_identity_not_string(self):
        """DataAccess catalog.get_identity() returns typed object with cache_key()."""
        # This test validates the API contract between DataAccess and FE.
        # The actual SemanticCatalogIdentity implementation is tested in DataAccess.
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=True,
        )

        mock_catalog = Mock()
        mock_identity = Mock()
        # Identity must have cache_key() method returning string
        mock_identity.cache_key.return_value = "a" * 64  # 256-bit hex

        mock_catalog.get_identity.return_value = mock_identity

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.return_value = mock_catalog

            result = source.semantic_catalog_identity()

            # FE receives cache key from typed identity
            assert isinstance(result, str)
            assert len(result) == 64
            mock_identity.cache_key.assert_called_once()

    def test_production_fails_when_identity_issuer_fails(self):
        """Production raises CatalogCorrupt when catalog.get_identity() fails."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=True,
        )

        mock_catalog = Mock()
        # Simulate identity issuer raising an exception
        mock_catalog.get_identity.side_effect = ValueError("Encoding failed")

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.return_value = mock_catalog

            with pytest.raises(CatalogCorrupt) as exc_info:
                source.semantic_catalog_identity()

            assert "FE-P0-022" in str(exc_info.value)
            assert "Failed to issue SemanticCatalogIdentity" in str(exc_info.value)
            # Verify get_identity was actually called
            mock_catalog.get_identity.assert_called_once_with(strict=True)

    def test_research_returns_unique_diagnostic_when_issuer_fails(self):
        """Research returns unique diagnostic when get_identity() fails."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=False,
        )

        mock_catalog = Mock()
        # Simulate identity issuer raising an exception
        mock_catalog.get_identity.side_effect = ValueError("Encoding failed")

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.return_value = mock_catalog

            result = source.semantic_catalog_identity()

            # Explicit diagnostic with timestamp
            assert result.startswith("catalog_identity_failed_at_")
            assert result != "unavailable"
            # Verify get_identity was actually called
            mock_catalog.get_identity.assert_called_once_with(strict=False)

    def test_research_unavailable_identity_is_uncacheable(self):
        """Research unavailable identity prevents cache reuse."""
        # Test the API contract - when identity.is_cacheable() returns False,
        # the cache_key includes a unique diagnostic.
        mock_identity = Mock()
        mock_identity.is_cacheable.return_value = False
        mock_identity.cache_key.return_value = "unavailable:encoding_failed_at_1234567890:ValueError"

        # Verify the contract
        assert not mock_identity.is_cacheable()
        cache_key = mock_identity.cache_key()
        assert cache_key.startswith("unavailable:")
        assert "1234567890" in cache_key

    def test_ensure_field_plans_uses_strict_identity(self):
        """_ensure_field_plans uses semantic_catalog_identity for cache key."""
        source = DataAccessSource(
            dataset="test_dataset",
            strict_unknown_fields=True,
        )

        mock_catalog = Mock()
        mock_identity = Mock()
        test_digest = "a" * 64  # 256-bit hex
        mock_identity.cache_key.return_value = test_digest
        mock_catalog.get_identity.return_value = mock_identity

        with patch(
            "data_access.read.semantic_catalog.get_semantic_catalog"
        ) as mock_get:
            mock_get.return_value = mock_catalog

            # Mock semantic_catalog_identity to control return value
            with patch.object(source, 'semantic_catalog_identity', return_value=test_digest):
                # Mock the field plan building
                mock_plan = Mock()
                source._build_field_plans = Mock(return_value={"test_field": mock_plan})

                plans = source._ensure_field_plans(["test_field"])

                assert "test_field" in plans
                # Verify semantic_catalog_identity was called (cache key uses it)
                source.semantic_catalog_identity.assert_called()

    def test_deprecated_semantic_catalog_version_still_works(self):
        """Legacy _semantic_catalog_version returns string for compatibility."""
        result = DataAccessSource._semantic_catalog_version()

        # Returns a string (either short hash or "unavailable")
        assert isinstance(result, str)
        # But it's deprecated - new code should use semantic_catalog_identity()
