#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R46 P0-X — the SINGLE authoritative GateEvidenceVerifier.

GateMatrix, VerificationManifest, and GateRunner all consume ONE verdict
system for gate truth.  This module IS that source of truth; nobody else
re-derives a gate status.  A PASS can only be produced here from real evidence.

Authoritative statuses: PASS / STALE / FAIL / BLOCKED / NOT_RUN / NOT_APPLICABLE.

Identity (P0-Y): a gate is compared by ``command_template_hash`` — the exact
pytest argv template.  ``test_scale_gates.py`` with ``-k 1k`` vs ``-k 100k``
are THREE distinct command identities (1K_SCALE / 10K_SCALE / 100K_SCALE), so
a gate is PASS only when ObservedCommandIdentitySet == ExpectedCommandIdentitySet
(missing / extra / duplicate evidence all keep the gate NOT a clean PASS).

NOT_APPLICABLE (R46 P0-29): a required gate is satisfied by PASS **or** a
legitimate NOT_APPLICABLE (with a reason/evidence).  SUBMODULE_REACHABILITY is
NOT_APPLICABLE with reason "monorepo contains no gitlinks" when the repo has no
mode-160000 git index entries.

LOCAL-ONLY: no git mutations, no network, no checkout/reset.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# Authoritative status vocabulary
PASS = "PASS"
STALE = "STALE"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
NOT_APPLICABLE = "NOT_APPLICABLE"

_ALL_STATUSES = frozenset({PASS, STALE, FAIL, BLOCKED, NOT_RUN, NOT_APPLICABLE})


@dataclass
class Verdict:
    """A single authoritative gate verdict + reasons (never silent)."""

    status: str = NOT_RUN
    tests: int = 0
    timestamp: str = "—"
    reasons: list[str] = field(default_factory=list)
    note: str = ""

    def reason_text(self) -> str:
        if self.reasons:
            return "; ".join(self.reasons)
        return self.note


class GateEvidenceVerifier:
    """Verify a gate's evidence into ONE authoritative Verdict.

    ``source_sha``    — git SHA the evidence was captured at (evidence["git_sha"])
    ``current_sha``   — live repo HEAD
    ``gitlinks_count``— number of mode-160000 git index entries (real submodules)
    """

    def __init__(
        self,
        source_sha: str | None = None,
        current_sha: str | None = None,
        gitlinks_count: int = 0,
    ) -> None:
        self.source_sha = source_sha
        self.current_sha = current_sha
        self.gitlinks_count = gitlinks_count

    # -- command identity (P0-Y) ----------------------------------------------
    @staticmethod
    def command_template_hash(command: list[str] | tuple[str, ...]) -> str:
        """Stable identity of a LOGICAL command (template argv as-written).

        Must match ``GateRunner._command_template_hash`` exactly: sha256 of the
        JSON-sorted argv, hashed BEFORE any ``{junitxml}`` substitution, so the
        same gate command yields the same hash on every run.
        """
        return hashlib.sha256(
            json.dumps(list(command), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def expected_command_identity_set(self, spec: Any | None) -> set[str]:
        """ExpectedCommandIdentitySet for a GateSpec: one hash per command.

        Verifier-backed gates (``spec.commands == ()``) have an empty expected
        set — their truth comes from the per-entry authoritative status instead.
        """
        if spec is None:
            return set()
        commands = tuple(getattr(spec, "commands", ()) or ())
        if not commands:
            return set()
        return {self.command_template_hash(list(argv)) for argv in commands}

    @staticmethod
    def observed_command_identity_set(entries: list[dict]) -> set[str]:
        """ObservedCommandIdentitySet from evidence entries (``command_hash``)."""
        return {
            str(e["command_hash"])
            for e in entries
            if isinstance(e, dict) and e.get("command_hash")
        }

    # -- main entry point -------------------------------------------------------
    def verify_gate(
        self,
        *,
        gate_name: str,
        spec: Any | None,
        entries: list[dict],
        live_command_hashes: set[str] | None = None,
    ) -> Verdict:
        """Return the authoritative verdict for one gate's evidence entries.

        ``live_command_hashes`` is the live ExpectedCommandIdentitySet (computed
        from the GateSpec at consume time); when omitted it is recomputed from
        ``spec`` here.  Passing it in lets the caller bind the check to the live
        spec it is rendering.
        """
        # 1) NOT_APPLICABLE — legitimate "nothing to prove" (P0-29).
        if gate_name == "SUBMODULE_REACHABILITY" and self.gitlinks_count == 0:
            return Verdict(
                status=NOT_APPLICABLE,
                note="monorepo contains no gitlinks",
                reasons=[
                    "no mode-160000 git index entries; nothing pinned to prove"
                ],
            )

        # 2) STALE — evidence was captured against a different source snapshot.
        if (
            self.source_sha is not None
            and self.current_sha is not None
            and self.source_sha != self.current_sha
        ):
            return Verdict(
                status=STALE,
                reasons=[
                    f"evidence SHA {self.source_sha} != HEAD {self.current_sha}"
                ],
            )

        # 3) NOT_RUN — no evidence at all.
        if not entries:
            return Verdict(status=NOT_RUN, note="no evidence recorded")

        # 4) Verifier-backed gates: truth comes from the authoritative
        #    per-entry status emitted by the runner's verifier (never re-derived).
        if spec is not None and getattr(spec, "verifier", None):
            return self._verify_verifier(gate_name, entries)

        # 5) Pytest gates: machine fields + exact command identity.
        return self._verify_pytest(
            gate_name, spec, entries, live_command_hashes
        )

    # -- verifier-backed gates ---------------------------------------------------
    def _verify_verifier(self, gate_name: str, entries: list[dict]) -> Verdict:
        statuses = [e.get("status") for e in entries if isinstance(e, dict)]
        notes = [
            str(e.get("note")) for e in entries
            if isinstance(e, dict) and e.get("note")
        ]
        reason = "; ".join(notes) if notes else f"{gate_name} verifier"
        if any(s == FAIL for s in statuses):
            return Verdict(FAIL, reasons=[f"{reason} reports FAIL"])
        if any(s == BLOCKED for s in statuses):
            return Verdict(BLOCKED, reasons=[f"{reason} reports BLOCKED"])
        if any(s == NOT_APPLICABLE for s in statuses):
            return Verdict(NOT_APPLICABLE, note=reason, reasons=notes)
        if all(s == PASS for s in statuses):
            return Verdict(PASS, note=reason, reasons=notes)
        return Verdict(NOT_RUN, reasons=[f"{reason}: no definitive status"])

    # -- pytest gates --------------------------------------------------------------
    def _verify_pytest(
        self,
        gate_name: str,
        spec: Any | None,
        entries: list[dict],
        live_command_hashes: set[str] | None,
    ) -> Verdict:
        expected = (
            live_command_hashes
            if live_command_hashes is not None
            else self.expected_command_identity_set(spec)
        )

        # Per-entry machine-field validation (VER-P0-01 honesty).
        ok: list[dict] = []
        total_tests = 0
        timestamps: list[str] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            cmd_hash = e.get("command_hash")
            exit_code = e.get("exit_code")
            tests = e.get("tests")
            failed = e.get("failed")
            errors = e.get("errors")
            ts = e.get("executed_at") or e.get("run_at") or ""
            if ts:
                timestamps.append(str(ts))

            if not cmd_hash or not isinstance(exit_code, int):
                return Verdict(
                    NOT_RUN,
                    reasons=[f"{gate_name}: evidence missing command_hash/exit_code"],
                )
            if exit_code != 0:
                return Verdict(
                    FAIL,
                    reasons=[f"{gate_name}: exit_code={exit_code} != 0"],
                )
            if failed:
                return Verdict(
                    FAIL,
                    reasons=[f"{gate_name}: failed={failed} > 0"],
                )
            if errors:
                return Verdict(
                    FAIL,
                    reasons=[f"{gate_name}: errors={errors} > 0"],
                )
            if not isinstance(tests, int) or tests <= 0:
                return Verdict(
                    BLOCKED,
                    reasons=[f"{gate_name}: nothing ran (tests={tests!r} <= 0)"],
                )
            total_tests += tests
            ok.append(e)

        timestamp = timestamps[0] if timestamps else "—"

        # Exact command identity (P0-Y): Observed == Expected.
        if expected:
            observed = self.observed_command_identity_set(ok)
            if len(ok) != len(observed):
                return Verdict(
                    NOT_RUN, tests=total_tests, timestamp=timestamp,
                    reasons=[f"{gate_name}: DUPLICATE command evidence"],
                )
            missing = expected - observed
            extra = observed - expected
            if missing:
                return Verdict(
                    NOT_RUN, tests=total_tests, timestamp=timestamp,
                    reasons=[f"{gate_name}: MISSING expected command(s) {sorted(missing)}"],
                )
            if extra:
                return Verdict(
                    NOT_RUN, tests=total_tests, timestamp=timestamp,
                    reasons=[f"{gate_name}: EXTRA unexpected command(s) {sorted(extra)}"],
                )
        elif len(ok) != len(entries):
            # Some entries were dropped; exact set cannot be proven.
            return Verdict(
                NOT_RUN, tests=total_tests, timestamp=timestamp,
                reasons=[f"{gate_name}: non-clean evidence entries present"],
            )

        # Skip policy (P0-27/28): a registered allowed-skip inventory is the
        # authoritative mechanism for declaring which skips are legitimate.
        if spec is not None:
            policy = getattr(spec, "skip_policy", "allowed")
            inv = getattr(spec, "allowed_skip_inventory", None)
            for o in ok:
                skipped = o.get("skipped") or 0
                if skipped <= 0:
                    continue
                if inv is not None:
                    # A per-test inventory was declared: a skip is legitimate only
                    # if this test path is registered in it.
                    if o.get("test") not in inv:
                        return Verdict(
                            NOT_RUN, tests=total_tests, timestamp=timestamp,
                            reasons=[
                                f"{gate_name}: skip on {o.get('test')} not in "
                                "allowed_skip_inventory",
                            ],
                        )
                elif policy == "fail_on_skip":
                    return Verdict(
                        NOT_RUN, tests=total_tests, timestamp=timestamp,
                        reasons=[f"{gate_name}: fail_on_skip with skipped>0"],
                    )

        return Verdict(
            PASS, tests=total_tests, timestamp=timestamp,
            reasons=[f"{gate_name}: {len(ok)} command(s) verified"],
        )


# ---------------------------------------------------------------------------
# Small helper for consumers that only need the expected command identity.
# ---------------------------------------------------------------------------
def expected_command_identity_set(spec: Any | None) -> set[str]:
    """Module-level helper: ExpectedCommandIdentitySet for a GateSpec."""
    return GateEvidenceVerifier().expected_command_identity_set(spec)


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    v = GateEvidenceVerifier()
    spec = type("S", (), {"commands": (("a.py", "-q"),), "tests": ("a.py",)})()
    print("expected:", v.expected_command_identity_set(spec))
    print("OK GateEvidenceVerifier import")
