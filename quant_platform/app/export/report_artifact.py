"""ReportExportArtifact — platform artifact for legacy report exports.

QRP-P10. The platform entity-izes "reports" as an artifact: a content snapshot
(``content``) plus a derived ``content_hash`` and provenance. This is the
contract layer for migrating the legacy report centers (``vectorbt_qs``
reporting and ``research_platform`` Campaign/Trial Ledger) onto the platform
artifact model — it deliberately does NOT talk to either domain.

PURE-STDLIB rule holds for the whole module (mirrors ``quant_platform.app.
contracts``): frozen dataclasses + enum + ``json`` for the serializability
verification and the DICT codec. No fastapi / pydantic / third-party imports.

Provenance
----------
- ``backtest_ref``: domain-native backtest identifier (e.g. a
  ``vectorbt_qs`` backtest id or a platform ``BacktestArtifactRef`` reference
  minus the signpost) — carried as an opaque string.
- ``campaign_ref``: domain-native ``research_platform`` Campaign/Trial Ledger
  reference — opaque string.
- ``producer``: one of ``ReportProducer`` values (``"vectorbt_qs"`` /
  ``"research_platform"`` / ``"platform"``).

``content`` is copied defensively on entry and verified JSON-serializable
fail-closed, so no non-serializable object can ever reach ``content_hash``.

Import rule: ``vectorbt_qs`` / ``research_platform`` internals are NOT imported
anywhere in this module (参照不 import). The only vectorbt-family adapter is
``ReportExporter.from_backtest_stats`` which consumes a decoded ``Mapping`` —
a caller that holds a pandas ``Series`` converts it with ``.to_dict()`` before
calling, so pandas/vectorbt never enter this module.
"""

from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from quant_platform.app.contracts._contenthash import canonical_str, content_hash
from quant_platform.app.contracts.artifact_ref import (
    ARTIFACT_TYPE_REPORT_EXPORT,
    ArtifactRef,
)
from quant_platform.app.storage.registry import ArtifactRegistry

__all__ = [
    "REPORT_KINDS",
    "REPORT_PRODUCERS",
    "ReportKind",
    "ReportProducer",
    "ReportExportArtifact",
    "ReportExporter",
    "report_to_dict",
    "from_dict",
    "serializable_content",
]


class ReportKind(enum.Enum):
    """Kind of exported report (backtest / campaign / health)."""

    BACKTEST = "backtest"
    CAMPAIGN = "campaign"
    HEALTH = "health"


REPORT_KINDS: frozenset[str] = frozenset(k.value for k in ReportKind)
"""All valid ``ReportExportArtifact.report_kind`` string values."""


class ReportProducer(enum.Enum):
    """Producer of an exported report artifact.

    Values are the producer strings mandated by the task —
    ``"vectorbt_qs"`` / ``"research_platform"`` / ``"platform"``.
    """

    VECTORBT_QS = "vectorbt_qs"
    RESEARCH_PLATFORM = "research_platform"
    PLATFORM = "platform"


REPORT_PRODUCERS: frozenset[str] = frozenset(e.value for e in ReportProducer)
"""All valid ``ReportExportArtifact.producer`` string values."""

_DEFAULT_PRODUCER = ReportProducer.PLATFORM
_DEFAULT_SCHEMA_VERSION = "1.0"


def serializable_content(content: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``content`` is JSON-serializable fail-closed; return a copy.

    Raises ``TypeError`` when any nested value cannot be JSON-serialized
    (nested ``dict`` / ``list`` / ``tuple`` / str / int / float / bool / None
    only; ``Enum`` values are allowed — they serialize as ``.value``, matching
    the ``_contenthash`` codec). The check is an explicit recursive walk, so a
    failure names the offending type deterministically and there is no
    ``str()``/``repr()`` fallback. ``content`` must be a ``Mapping``.
    """
    if not isinstance(content, Mapping):
        raise TypeError(
            f"report content must be a mapping, got {type(content).__name__}"
        )

    def _check(value: Any) -> None:
        if isinstance(value, enum.Enum):
            _check(value.value)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                _check(key)
                _check(item)
            return
        if isinstance(value, (list, tuple)):
            for index in range(value.__len__()):
                _check(value[index])
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (str, int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise TypeError(
                    "report content must contain only finite floats; "
                    f"got {value!r}"
                )
            return
        if value is None:
            return
        raise TypeError(
            f"report content is not JSON-serializable: "
            f"unsupported type {type(value).__name__} (value={value!r})"
        )

    def _copy(value: Any) -> Any:
        """Deep copy so the returned snapshot never aliases the caller's dicts."""
        if isinstance(value, enum.Enum):
            return _copy(value.value)
        if isinstance(value, dict):
            return {k: _copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_copy(item) for item in value)
        return value

    _check(content)
    return _copy(content)


@dataclass(frozen=True, init=False)
class ReportExportArtifact:
    """Immutable report-export artifact (content snapshot + provenance).

    Fields
    ------
    - ``artifact_type = "report_export"`` (``ARTIFACT_TYPE_REPORT_EXPORT``),
      constant — never user-settable.
    - ``report_kind``: ``"backtest"`` / ``"campaign"`` / ``"health"``.
    - ``backtest_ref`` / ``campaign_ref``: optional opaque domain refs
      (whichever applies; for ``health`` both may be ``None``).
    - ``content``: dict report content snapshot (validated JSON-serializable).
    - ``content_hash``: **derived-only** — sha256 over the canonical semantic
      fields (``contracts._contenthash.content_hash``), computed automatically
      on construction; any supplied value is replaced.
    - ``exported_at``: timezone-aware UTC ISO-8601.
    - ``producer``: ``"vectorbt_qs"`` / ``"research_platform"`` / ``"platform"``.

    ``content_hash`` is authoritative = ``contracts/_contenthash.py``; this
    module adds NO new hash logic (a caller-supplied hash from a stale DICT
    round-trip is overwritten — the hash is never trusted from the wire).
    """

    artifact_type: str = field(
        default=ARTIFACT_TYPE_REPORT_EXPORT, init=False, repr=False
    )
    report_kind: str = "backtest"
    backtest_ref: str | None = None
    campaign_ref: str | None = None
    content: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False, repr=False)
    exported_at: str = ""
    producer: str = field(default=_DEFAULT_PRODUCER.value)

    def __init__(
        self,
        report_kind: str = "backtest",
        backtest_ref: str | None = None,
        campaign_ref: str | None = None,
        content: Mapping[str, Any] | None = None,
        *,
        exported_at: str = "",
        producer: str = _DEFAULT_PRODUCER.value,
    ) -> None:
        """Construct, validating + deriving ``content_hash``.

        ``artifact_type`` is always ``"report_export"`` (constant) and
        ``content_hash`` is always derived — a caller can never set either.
        ``content`` defaults to ``{}``.
        """
        if report_kind not in REPORT_KINDS:
            raise ValueError(
                f"unknown report_kind: {report_kind!r} "
                f"(expected one of {sorted(REPORT_KINDS)})"
            )
        if producer not in REPORT_PRODUCERS:
            raise ValueError(
                f"unknown producer: {producer!r} "
                f"(expected one of {sorted(REPORT_PRODUCERS)})"
            )
        if not backtest_ref and not campaign_ref:
            if report_kind != "health":
                raise ValueError(
                    f"report_kind={report_kind!r} requires a backtest_ref "
                    f"or campaign_ref reference"
                )
        if isinstance(backtest_ref, str) and not backtest_ref.strip():
            raise ValueError("backtest_ref must be a non-empty string if provided")
        if isinstance(campaign_ref, str) and not campaign_ref.strip():
            raise ValueError("campaign_ref must be a non-empty string if provided")
        # JSON-serializable fail-close (before any hashing).
        content = serializable_content({} if content is None else content)
        if exported_at:
            _require_aware_datetime(exported_at)
        else:
            exported_at = datetime.now(tz=timezone.utc).isoformat()
        object.__setattr__(self, "report_kind", report_kind)
        object.__setattr__(self, "backtest_ref", backtest_ref)
        object.__setattr__(self, "campaign_ref", campaign_ref)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "exported_at", exported_at)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(
            self,
            "content_hash",
            content_hash(
                ARTIFACT_TYPE_REPORT_EXPORT,
                report_kind,
                backtest_ref,
                campaign_ref,
                content,
                exported_at,
                producer,
            ),
        )

    @property
    def semantic_hash(self) -> str:
        """Derived semantic hash — all semantic fields except ``exported_at``.

        Identical content exported at a different time keeps the same semantic
        identity while its ``content_hash`` differs.
        """
        return content_hash(
            ARTIFACT_TYPE_REPORT_EXPORT,
            self.report_kind,
            self.backtest_ref,
            self.campaign_ref,
            self.content,
            self.producer,
        )


class ReportExporter:
    """Export report content snapshots as ``ReportExportArtifact`` artifacts.

    The exporter owns nothing persistent — ``to_registry`` delegates the actual
    registration to :class:`quant_platform.app.storage.registry.ArtifactRegistry`
    (the registry re-validates artifact types and rejects unknown ones
    fail-closed with ``UnknownArtifactTypeError``).
    """

    __slots__ = ()

    def export(
        self,
        refs: Mapping[str, str] | None = None,
        content: Mapping[str, Any] | None = None,
        kind: str = "backtest",
        *,
        producer: str = _DEFAULT_PRODUCER.value,
        exported_at: str = "",
    ) -> ReportExportArtifact:
        """Build a ``ReportExportArtifact`` from a plain mapping.

        ``refs`` keys: ``"backtest"`` / ``"campaign"`` (both optional). Raises
        on empty refs for kinds that need one, and on unknown kind / producer
        via the artifact's own validation.
        """
        refs = dict(refs or {})
        return ReportExportArtifact(
            report_kind=kind,
            backtest_ref=refs.get("backtest"),
            campaign_ref=refs.get("campaign"),
            content={} if content is None else content,
            exported_at=exported_at,
            producer=producer,
        )

    def from_backtest_stats(
        self,
        stats: Mapping[str, Any] | None,
        *,
        backtest_ref: str | None = None,
        campaign_ref: str | None = None,
        producer: str = _DEFAULT_PRODUCER.value,
        exported_at: str = "",
    ) -> ReportExportArtifact:
        """Adapter from a decoded backtest stats mapping (report posture).

        ``BacktestArtifactRef`` treats report content as a metadata dict.
        ``vectorbt_qs``'s ``portfolio_report(pf)`` returns a ``pandas.Series``;
        a caller exporting that converts it with ``.to_dict()`` *before*
        calling this, so pandas/vectorbt are never imported here.
        """
        if not backtest_ref and not campaign_ref:
            raise ValueError(
                "from_backtest_stats requires at least backtest_ref or "
                "campaign_ref"
            )
        return ReportExportArtifact(
            report_kind=ReportKind.BACKTEST.value,
            backtest_ref=backtest_ref,
            campaign_ref=campaign_ref,
            content={} if stats is None else stats,
            exported_at=exported_at,
            producer=producer,
        )

    def to_registry(
        self, artifact: ReportExportArtifact, registry: ArtifactRegistry
    ) -> ArtifactRef:
        """Register ``artifact`` in ``registry`` as a REPORT_EXPORT artifact.

        The storage ``ArtifactRef`` is built with the producer set to the
        artifact's producer and ``producer_source_ref`` carrying the underlying
        domain reference. Unknown artifact types are rejected inside the
        registry (fail-closed).
        """
        if artifact.artifact_type != ARTIFACT_TYPE_REPORT_EXPORT:
            raise ValueError(
                f"ReportExporter.to_registry expects REPORT_EXPORT artifacts, "
                f"got {artifact.artifact_type!r}"
            )
        source_ref = artifact.campaign_ref or artifact.backtest_ref or "health"
        artifact_id = f"report:{source_ref}:{artifact.report_kind}"
        storage_uri = (
            f"cos://artifacts/report-export/{source_ref}/{artifact.report_kind}"
        )
        ref = ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact.artifact_type,
            schema_version=_DEFAULT_SCHEMA_VERSION,
            content_hash=artifact.content_hash,
            storage_uri=storage_uri,
            size_bytes=_json_size_bytes(artifact.content),
            created_at=_parse_exported_at(artifact.exported_at),
            producer_type=artifact.producer,
            producer_version="1.0",
            media_type="application/json",
            producer_source_ref=source_ref,
            semantic_hash=artifact.semantic_hash,
        )
        return registry.register(ref)


# ---- DICT codec -----------------------------------------------------------


def _json_default(value: Any) -> Any:
    """``json.dumps`` default that emits Enum values as their ``.value``.

    ``serializable_content`` guarantees no other non-JSON type reaches here.
    """
    if isinstance(value, enum.Enum):
        return value.value
    raise TypeError(f"unexpected non-JSON value in report: {value!r}")


def report_to_dict(
    artifact: ReportExportArtifact, serializer: Any = json
) -> dict[str, Any]:
    """DICT snapshot of a report artifact.

    Uses the given ``serializer`` for encoding (default stdlib ``json``) and
    re-parses through ``json.loads`` so the returned value is always a plain
    nested dict regardless of the serializer. ``content_hash`` is included for
    round-trip verification; ``from_dict`` always recomputes it and ignores /
    rejects a stale copy.
    """
    data: dict[str, Any] = {}
    for field_name in _REPORT_FIELD_ORDER:
        value = getattr(artifact, field_name)
        if isinstance(value, enum.Enum):
            value = value.value
        data[field_name] = value
    return json.loads(serializer.dumps(data, default=_json_default))


def from_dict(
    data: Mapping[str, Any],
    *,
    kind: str = "backtest",
    backtest_ref: str | None = None,
    campaign_ref: str | None = None,
    producer: str = _DEFAULT_PRODUCER.value,
    exported_at: str = "",
) -> ReportExportArtifact:
    """Rehydrate a ``ReportExportArtifact`` from its DICT snapshot.

    Report kind / producer / refs are treated as *caller context*, never read
    verbatim from the wire: ``kind``, ``backtest_ref``, ``campaign_ref``,
    ``producer`` and ``exported_at`` are passed explicitly (defaulting to the
    snapshot values where the snapshot carries them, except ``kind`` which is
    always caller-supplied).

    Legacy round-trip compatibility:
      - old dicts *without* a ``content_hash`` key are accepted — the hash is
        recomputed (derived-only) on construction;
      - a stale ``content_hash`` key carried over from a previous round-trip
        must match the recomputed value — a mismatch raises ``ValueError`` so
        tampered content is detected fail-closed.
    """
    data = dict(data)
    if data.get("artifact_type", ARTIFACT_TYPE_REPORT_EXPORT) != ARTIFACT_TYPE_REPORT_EXPORT:
        raise ValueError(
            f"from_dict requires artifact_type={ARTIFACT_TYPE_REPORT_EXPORT!r}, "
            f"got {data.get('artifact_type')!r}"
        )
    legacy_hash = data.get("content_hash")
    artifact = ReportExportArtifact(
        report_kind=kind,
        backtest_ref=backtest_ref or data.get("backtest_ref"),
        campaign_ref=campaign_ref or data.get("campaign_ref"),
        content=data.get("content", {}),
        exported_at=exported_at or data.get("exported_at", ""),
        producer=producer or data.get("producer", _DEFAULT_PRODUCER.value),
    )
    if legacy_hash is not None:
        if legacy_hash != artifact.content_hash:
            raise ValueError(
                f"stale content_hash in dict snapshot: got {legacy_hash!r}, "
                f"recomputed {artifact.content_hash!r} — content was tampered "
                f"after export"
            )
    return artifact


# ---- private helpers --------------------------------------------------------


def _require_aware_datetime(value: str) -> datetime:
    """Parse ``value`` as ISO-8601; require timezone-awareness (like the
    canonical content-hash codec, naive datetimes are rejected)."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(
            f"exported_at must be timezone-aware ISO-8601, got {value!r}"
        )
    return dt


def _parse_exported_at(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _json_size_bytes(content: Mapping[str, Any]) -> int:
    text = json.dumps(dict(content), sort_keys=True, default=_json_default)
    return len(text.encode("utf-8"))


_REPORT_FIELD_ORDER = (
    "artifact_type",
    "report_kind",
    "backtest_ref",
    "campaign_ref",
    "content",
    "content_hash",
    "exported_at",
    "producer",
)