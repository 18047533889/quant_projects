"""DA2-P0-002 semantic catalog correctness-identity regressions."""
from __future__ import annotations

from data_access.read.query_cache import query_cache_key
from data_access.read.semantic_catalog import SemanticField, SemanticFieldCatalog


def _catalog(latency: int | None) -> SemanticFieldCatalog:
    field = SemanticField(
        logical_name="earnings_event",
        dataset="events",
        physical_name="earnings_event",
        availability_latency=latency,
    )
    return SemanticFieldCatalog({"earnings_event": field})


def _query_key(catalog_digest: str) -> str:
    return query_cache_key(
        dataset="events",
        params={},
        time_range=None,
        instruments=None,
        columns=None,
        manifest_token=None,
        catalog_fingerprint=catalog_digest,
    )


def test_availability_latency_changes_catalog_and_query_identity() -> None:
    immediate = _catalog(None)
    delayed = _catalog(1)

    assert immediate.to_dict()["earnings_event"]["availability_latency"] is None
    assert delayed.to_dict()["earnings_event"]["availability_latency"] == 1

    immediate_identity = immediate.get_identity(strict=True)
    delayed_identity = delayed.get_identity(strict=True)
    assert immediate_identity.available is True
    assert delayed_identity.available is True
    assert len(immediate_identity.digest) == 64
    assert len(delayed_identity.digest) == 64
    assert immediate_identity.digest != delayed_identity.digest

    assert _query_key(immediate_identity.digest) != _query_key(delayed_identity.digest)
