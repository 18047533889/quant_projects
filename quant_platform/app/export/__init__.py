"""Platform export package — report artifact contract layer (QRP-P10).

Entity-izes legacy report outputs (``vectorbt_qs`` reporting /
``research_platform`` Campaign/Trial Ledger) as platform artifacts:

- :mod:`quant_platform.app.export.report_artifact` — ``ReportExportArtifact``
  (immutable content snapshot + derived ``content_hash`` + provenance),
  ``ReportExporter`` (``export`` / ``from_backtest_stats`` / ``to_registry``)
  and the ``report_to_dict`` / ``from_dict`` DICT codec with legacy hash
  recompute.

PURE-STDLIB rule holds here (mirrors ``app/contracts``). Domain packages
(``vectorbt_qs`` / ``research_platform``) are referenced only for provenance
strings, never imported.
"""

from __future__ import annotations

from .report_artifact import (
    REPORT_KINDS,
    REPORT_PRODUCERS,
    ReportExportArtifact,
    ReportExporter,
    ReportKind,
    ReportProducer,
    from_dict,
    report_to_dict,
    serializable_content,
)

__all__ = [
    "ReportExportArtifact",
    "ReportExporter",
    "ReportKind",
    "ReportProducer",
    "REPORT_KINDS",
    "REPORT_PRODUCERS",
    "report_to_dict",
    "from_dict",
    "serializable_content",
]
