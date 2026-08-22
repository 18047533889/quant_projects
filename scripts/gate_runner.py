#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VER-P0-01 — honesty-first gate runner.

Replaces the old "write result='passed' from config" pattern.  A PASS can now
ONLY be produced by ACTUALLY executing the pytest command and reading back a
structured result (JUnit XML).  Every evidence entry carries machine fields
(command_hash / exit_code / passed / failed / errors / skipped / duration /
executed_at) so a downstream consumer (gen_verification_manifest.py) can never
mistake a configured count for a real run.

Status derivation (strict, no shortcuts):

    PASS     exit_code == 0 AND junit.failures == 0 AND junit.errors == 0
             AND junit.tests > 0            (empty run is never a PASS)
    FAIL     any test failure / error / non-zero exit / timeout
    BLOCKED  nothing actually ran: junit missing AND no tests collected,
             or junit.tests == 0

A gate that points at a non-existent test file, or whose command could not
execute, is BLOCKED/FAIL — never PASS.

LOCAL-ONLY: no git mutations, no network, no checkout/reset.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "evidence"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


class Status:
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


# ---------------------------------------------------------------------------
# Gate specification
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GateSpec:
    """A release gate = one or more exact pytest command argv lists.

    ``commands`` entries are the exact pytest argument lists (the interpreter
    is prepended by the runner).  Each entry SHOULD contain a ``--junitxml=
    {junitxml}`` token; the runner replaces ``{junitxml}`` with a unique
    absolute path so concurrent commands never collide.  ``tests`` holds the
    test file path(s) (relative to REPO_ROOT), aligned 1:1 with ``commands``.
    """

    gate_id: str
    commands: tuple[tuple[str, ...], ...]
    tests: tuple[str, ...]
    timeout_sec: int = 600
    env: dict | None = None
    skip_policy: str = "allowed"  # "allowed" | "fail_on_skip"
    failure_policy: str = "strict"  # any failure/error => gate FAIL


# QE tests importable source is build/lib/quant_evaluator (the repo-root
# quant_evaluator/ dir is a near-empty stub that shadows it under pytest).
QE_ENV = {"PYTHONPATH": str(REPO_ROOT / "quant_evaluator" / "build" / "lib")}
FP_PREPROCESS_ENV = {"PYTHONPATH": f"{REPO_ROOT / 'factor_preprocess' / 'build' / 'lib'}:{REPO_ROOT / 'quant_evaluator' / 'build' / 'lib'}"}

_JUNIT_TOK = "--junitxml={junitxml}"

GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(
        gate_id="NUMERICAL_ORACLE",
        commands=(
            ("quant_evaluator/tests/test_numerical_oracle.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("quant_evaluator/tests/test_numerical_oracle.py",),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="PROPERTY",
        commands=(
            ("quant_evaluator/tests/test_metamorphic.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("quant_evaluator/tests/test_consistency.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("quant_evaluator/tests/test_metamorphic.py", "quant_evaluator/tests/test_consistency.py"),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="CROSS_PACKAGE",
        commands=(
            ("integration_tests/test_cross_package_contracts.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("integration_tests/test_cross_package_contracts.py",),
    ),
    GateSpec(
        gate_id="ASHARE_SEMANTIC_CONTRACT_GOLDEN",
        commands=(
            ("integration_tests/test_ashare_semantic_golden.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("integration_tests/test_ashare_semantic_golden.py",),
    ),
    GateSpec(
        gate_id="ASHARE_REAL_DATA_SHADOW",
        commands=(
            ("integration_tests/test_real_ashare_shadow.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("integration_tests/test_real_ashare_shadow.py",),
        skip_policy="fail_on_skip",  # a skip is never a PASS for the real-data shadow
    ),
    GateSpec(
        gate_id="SERIALIZATION",
        commands=(
            ("quant_evaluator/tests/test_qe_serialization.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_assets/tests/registry/test_serialization_codec.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=(
            "quant_evaluator/tests/test_qe_serialization.py",
            "factor_assets/tests/registry/test_serialization_codec.py",
        ),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="CHECKPOINT_RESUME",
        commands=(
            ("factor_engine/tests/runtime/test_r10_stateful_checkpoint_2026_08.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/runtime/test_r10_stateful_checkpoint_2026_08.py",),
    ),
    GateSpec(
        gate_id="UNIT",
        commands=(
            ("factor_engine/tests/test_capability_registry.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_engine/tests/test_concurrency_safety.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/test_capability_registry.py", "factor_engine/tests/test_concurrency_safety.py"),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="LEAKAGE",
        commands=(
            ("factor_engine/tests/modeling/test_evaluation_leakage_evidence.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_preprocess/tests/contracts/test_leakage_properties.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/modeling/test_evaluation_leakage_evidence.py", "factor_preprocess/tests/contracts/test_leakage_properties.py"),
        env=FP_PREPROCESS_ENV,
    ),
    GateSpec(
        gate_id="PIT",
        commands=(
            ("factor_engine/tests/runtime/test_pit_audit.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_engine/tests/runtime/test_label_pit.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_engine/tests/runtime/test_r10_pit_tristate.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/runtime/test_pit_audit.py", "factor_engine/tests/runtime/test_label_pit.py", "factor_engine/tests/runtime/test_r10_pit_tristate.py"),
    ),
    GateSpec(
        gate_id="DETERMINISM",
        commands=(
            ("factor_engine/tests/test_comprehensive_data_consistency.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_engine/tests/backend/test_r21_def5_payload_hash.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/test_comprehensive_data_consistency.py", "factor_engine/tests/backend/test_r21_def5_payload_hash.py"),
    ),
    GateSpec(
        gate_id="FRESH_WHEEL",
        commands=(
            ("factor_engine/scripts/wheel_clean_install_smoke.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/scripts/wheel_clean_install_smoke.py",),
    ),
    GateSpec(
        gate_id="1K_SCALE",
        commands=(
            ("quant_evaluator/tests/test_scale_gates.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider", "-k", "1k or 1K or scale_1k"),
        ),
        tests=("quant_evaluator/tests/test_scale_gates.py",),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="10K_SCALE",
        commands=(
            ("quant_evaluator/tests/test_scale_gates.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider", "-k", "10k or 10K or scale_10k"),
        ),
        tests=("quant_evaluator/tests/test_scale_gates.py",),
        env=QE_ENV,
    ),
    GateSpec(
        gate_id="100K_SCALE",
        commands=(
            ("quant_evaluator/tests/test_scale_gates.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider", "-k", "100k or 100K or scale_100k"),
        ),
        tests=("quant_evaluator/tests/test_scale_gates.py",),
        env=QE_ENV,
    ),
)


def find_spec(gate_id: str) -> GateSpec | None:
    for spec in GATE_SPECS:
        if spec.gate_id == gate_id:
            return spec
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class GateRunner:
    def __init__(self, python: Path | str = VENV_PYTHON, cwd: Path = REPO_ROOT):
        self.python = str(python)
        self.cwd = str(cwd)

    # -- helpers ------------------------------------------------------------
    def command_hash(self, argv: list[str]) -> str:
        return hashlib.sha256(json.dumps(argv, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _git_sha() -> str | None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _parse_junit(xml_path: Path) -> dict | None:
        """Parse a pytest JUnit XML file into honest counts (or None)."""
        if xml_path is None or not Path(xml_path).is_file():
            return None
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return None
        if root.tag == "testsuites":
            suites = root.findall("testsuite")
        elif root.tag == "testsuite":
            suites = [root]
        else:
            return None
        tests = failures = errors = skipped = 0
        xfailed = xpassed = 0
        for suite in suites:
            tests += int(suite.attrib.get("tests", 0) or 0)
            failures += int(suite.attrib.get("failures", 0) or 0)
            errors += int(suite.attrib.get("errors", 0) or 0)
            skipped += int(suite.attrib.get("skipped", 0) or 0)
            for tc in suite.iter("testcase"):
                for child in tc:
                    if child.tag == "skipped":
                        typ = (child.attrib.get("type", "") or "").lower()
                        if "xpass" in typ:
                            xpassed += 1
                        elif "xfail" in typ:
                            xfailed += 1
        return {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "xfailed": xfailed,
            "xpassed": xpassed,
        }

    def _collect_count(self, argv: list[str], spec: GateSpec, timeout: int) -> int | None:
        """Fallback: count how many tests pytest *would* collect (no execution).

        This is ONLY used when JUnit XML was not produced.  Collection is not
        execution, so a collected count can never upgrade a gate to PASS.
        """
        clean = [a for a in argv if "--junitxml" not in a]
        cmd = [self.python, "-m", "pytest", *clean, "--co", "-q", "-p", "no:cacheprovider"]
        env = dict(os.environ)
        if spec.env:
            env.update(spec.env)
        try:
            proc = subprocess.run(
                cmd, cwd=self.cwd, env=env, capture_output=True,
                text=True, timeout=min(timeout, 120),
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            return None
        m = re.search(r"(\d+) tests? collected", out)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _derive_status(
        exit_code: int | None,
        junit: dict | None,
        collected: int | None,
        spec: GateSpec,
        timed_out: bool,
    ) -> str:
        if timed_out:
            return Status.FAIL
        if junit is None:
            # No structured result: we cannot prove a pass.  If nothing was
            # even collected, nothing ran -> BLOCKED.
            n = collected if collected is not None else 0
            return Status.BLOCKED if n == 0 else Status.FAIL
        tests = junit["tests"]
        if tests == 0:
            return Status.BLOCKED
        if exit_code == 0 and junit["failures"] == 0 and junit["errors"] == 0:
            if spec.skip_policy == "fail_on_skip" and junit["skipped"] > 0:
                return Status.FAIL
            return Status.PASS
        return Status.FAIL

    # -- public API ----------------------------------------------------------
    def run_command(
        self,
        argv: tuple[str, ...],
        spec: GateSpec,
        timeout: int | None = None,
    ) -> dict:
        timeout = timeout or spec.timeout_sec
        rendered = list(argv)
        xml_path: Path | None = None
        for idx, tok in enumerate(rendered):
            if "{junitxml}" in tok:
                xml_path = Path(tempfile.gettempdir()) / (
                    f"gate_runner_{spec.gate_id}_{int(time.time() * 1_000_000)}.xml"
                )
                rendered[idx] = tok.replace("{junitxml}", str(xml_path))
        full_cmd = [self.python, "-m", "pytest", *rendered]
        cmd_hash = self.command_hash(full_cmd)
        env = dict(os.environ)
        if spec.env:
            env.update(spec.env)

        started = time.time()
        timed_out = False
        exit_code: int | None = None
        try:
            proc = subprocess.run(
                full_cmd, cwd=self.cwd, env=env, capture_output=True,
                text=True, timeout=timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
        duration = round(time.time() - started, 3)

        junit = self._parse_junit(xml_path) if xml_path is not None else None
        collected: int | None = None
        if junit is None:
            collected = self._collect_count(rendered, spec, timeout)

        status = self._derive_status(exit_code, junit, collected, spec, timed_out)

        counts = {
            "passed": None,
            "failed": None,
            "errors": None,
            "skipped": None,
            "xfailed": None,
            "xpassed": None,
        }
        if junit is not None:
            counts["passed"] = max(
                0, junit["tests"] - junit["failures"] - junit["errors"] - junit["skipped"]
            )
            counts["failed"] = junit["failures"]
            counts["errors"] = junit["errors"]
            counts["skipped"] = junit["skipped"]
            counts["xfailed"] = junit["xfailed"]
            counts["xpassed"] = junit["xpassed"]

        entry: dict = {
            "test": "",
            "argv": full_cmd,
            "command_hash": cmd_hash,
            "exit_code": exit_code,
            "status": status,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "xfailed": counts["xfailed"],
            "xpassed": counts["xpassed"],
            "tests": junit["tests"] if junit is not None else None,
            "collected_tests": collected,
            "duration_sec": duration,
            "executed_at": self._now_iso(),
            "timed_out": timed_out,
            "note": "TIMEOUT" if timed_out else None,
        }
        return entry

    def run_gate(
        self,
        spec: GateSpec,
        timeout: int | None = None,
        tests: tuple[str, ...] | None = None,
    ) -> dict:
        """Run every command in the gate; PASS only if ALL commands PASS."""
        entries: list[dict] = []
        tests = tests if tests is not None else spec.tests
        for i, argv in enumerate(spec.commands):
            entry = self.run_command(argv, spec, timeout=timeout)
            entry["test"] = tests[i] if i < len(tests) else ""
            entries.append(entry)
        if all(e["status"] == Status.PASS for e in entries):
            gate_status = Status.PASS
        elif any(e["status"] == Status.FAIL for e in entries):
            gate_status = Status.FAIL
        elif any(e["status"] == Status.BLOCKED for e in entries):
            gate_status = Status.BLOCKED
        else:
            gate_status = Status.NOT_RUN
        return {"name": spec.gate_id, "status": gate_status, "evidence": entries}

    def build_artifact(self, specs: list[GateSpec], timeout: int | None = None) -> dict:
        gates: list[dict] = []
        for spec in specs:
            missing = [t for t in spec.tests if not (REPO_ROOT / t).is_file()]
            if missing:
                print(
                    f"  [skip] gate {spec.gate_id}: missing test files {missing} -> NOT_RUN",
                    file=__import__("sys").stderr,
                )
                gates.append({
                    "name": spec.gate_id,
                    "status": Status.NOT_RUN,
                    "evidence": [],
                    "note": f"test file missing in current tree: {missing}",
                })
                continue
            result = self.run_gate(spec, timeout=timeout)
            gates.append(result)
            ev = result["evidence"]
            if ev:
                detail = ", ".join(
                    f"{e['test']}: exit={e['exit_code']} "
                    f"passed={e['passed']} failed={e['failed']} errors={e['errors']} "
                    f"skipped={e['skipped']}"
                    for e in ev
                )
            else:
                detail = "(no evidence)"
            print(f"  gate {spec.gate_id:20s} -> {result['status']:8s} {detail}")
        return {
            "schema_version": 2,
            "generated_by": "gate_runner",
            "git_sha": self._git_sha(),
            "timestamp": self._now_iso(),
            "gates": gates,
        }


if __name__ == "__main__":  # pragma: no cover - manual CLI
    import sys
    runner = GateRunner()
    artifact = runner.build_artifact(list(GATE_SPECS))
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
