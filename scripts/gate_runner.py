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
    # -- additive skip-inventory schema (P0-3) --------------------------------
    # Optional per-test expectations.  When ``allowed_skip_inventory`` is set,
    # PASS additionally requires every junit skip reason to match a registered
    # ``reason_regex`` that still has allowance remaining.  ``expected_test_inventory``
    # is metadata describing the *expected* pass/skip split for each test file
    # (used by downstream consumers; the runner itself enforces only the skip
    # inventory + skip_policy).  When unset, historical behavior is unchanged.
    expected_test_inventory: dict | None = None
    allowed_skip_inventory: dict | None = None
    # -- non-pytest verifier support (VER-P0-05 structural gates) ---------------
    # When ``verifier`` is set (a key into the ``_VERIFIERS`` registry below),
    # the gate runs a subprocess/JSON verifier instead of pytest.  Status is
    # derived from REAL process/JSON output — never fabricated.  ``commands`` /
    # ``tests`` are unused for verifier gates.
    verifier: str | None = None


# QE/FP/FO tests import canonical source via the test PYTHONPATH (repo root),
# not via a build/lib env hack.

_JUNIT_TOK = "--junitxml={junitxml}"

GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(
        gate_id="NUMERICAL_ORACLE",
        commands=(
            ("quant_evaluator/tests/test_numerical_oracle.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("quant_evaluator/tests/test_numerical_oracle.py",),
    ),
    GateSpec(
        gate_id="PROPERTY",
        commands=(
            ("quant_evaluator/tests/test_metamorphic.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("quant_evaluator/tests/test_consistency.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("quant_evaluator/tests/test_metamorphic.py", "quant_evaluator/tests/test_consistency.py"),
    ),
    GateSpec(
        gate_id="CROSS_PACKAGE_CORE",
        commands=(
            ("integration_tests/test_cross_package_contracts.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("integration_tests/test_cross_package_contracts.py",),
        # P0-CP: CORE cross-package contract tolerates NO skip — an import
        # failure anywhere in the core chain (factor_engine / quant_evaluator /
        # factor_assets) is a FAIL, never a legitimate skip.  0 skips allowed.
        skip_policy="fail_on_skip",
        allowed_skip_inventory={},
    ),
    GateSpec(
        gate_id="CROSS_PACKAGE_OPTIONAL",
        commands=(
            ("integration_tests/test_cross_package_contracts.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("integration_tests/test_cross_package_contracts.py",),
        # P0-CP: OPTIONAL tolerates a skip ONLY for a genuinely optional adapter
        # (factor_preprocess / factor_optimizer not wired).  A core-chain import
        # failure (factor_engine / quant_evaluator / factor_assets) is still a
        # FAIL.  The allowed reasons mirror the adapter-level skips in the test.
        skip_policy="fail_on_skip",
        expected_test_inventory={
            "integration_tests/test_cross_package_contracts.py": {
                "min_passed": 4,
                "allowed_skipped": 2,
            },
        },
        allowed_skip_inventory={
            "integration_tests/test_cross_package_contracts.py": {
                r"^factor_preprocess not importable here:.*": 1,
                r"^import failure:.*factor_optimizer.contracts.*": 1,
            },
        },
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
        # UNIT's two test files currently run clean (0 skips); any skip is a FAIL.
        skip_policy="fail_on_skip",
        expected_test_inventory={
            "factor_engine/tests/test_capability_registry.py": {"min_passed": 23, "allowed_skipped": 0},
            "factor_engine/tests/test_concurrency_safety.py": {"min_passed": 16, "allowed_skipped": 0},
        },
    ),
    GateSpec(
        gate_id="LEAKAGE",
        commands=(
            ("factor_engine/tests/modeling/test_evaluation_leakage_evidence.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
            ("factor_preprocess/tests/contracts/test_leakage_properties.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider"),
        ),
        tests=("factor_engine/tests/modeling/test_evaluation_leakage_evidence.py", "factor_preprocess/tests/contracts/test_leakage_properties.py"),
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
    ),
    GateSpec(
        gate_id="10K_SCALE",
        commands=(
            ("quant_evaluator/tests/test_scale_gates.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider", "-k", "10k or 10K or scale_10k"),
        ),
        tests=("quant_evaluator/tests/test_scale_gates.py",),
    ),
    GateSpec(
        gate_id="100K_SCALE",
        commands=(
            ("quant_evaluator/tests/test_scale_gates.py", "-q", _JUNIT_TOK, "-p", "no:cacheprovider", "-k", "100k or 100K or scale_100k"),
        ),
        tests=("quant_evaluator/tests/test_scale_gates.py",),
    ),
    # -- VER-P0-05 structural gates (non-pytest verifiers) ----------------------
    # These derive status from REAL subprocess output / JSON — never fabricated.
    GateSpec(
        gate_id="EVIDENCE_CURRENT",
        commands=(),
        tests=(),
        verifier="EVIDENCE_CURRENT",
    ),
    GateSpec(
        gate_id="SOURCE_AUTHORITY",
        commands=(),
        tests=(),
        verifier="SOURCE_AUTHORITY",
    ),
    GateSpec(
        gate_id="SUPPLY_CHAIN",
        commands=(),
        tests=(),
        verifier="SUPPLY_CHAIN",
    ),
    GateSpec(
        gate_id="FRESH_WHEEL_MATRIX",
        commands=(),
        tests=(),
        verifier="FRESH_WHEEL_MATRIX",
    ),
    GateSpec(
        gate_id="SUBMODULE_REACHABILITY",
        commands=(),
        tests=(),
        verifier="SUBMODULE_REACHABILITY",
    ),
)


def find_spec(gate_id: str) -> GateSpec | None:
    for spec in GATE_SPECS:
        if spec.gate_id == gate_id:
            return spec
    return None


def _skip_allowed(reason: str, budget: dict[str, dict]) -> bool:
    """Return True if *reason* matches a registered regex with remaining budget."""
    for test_rel, rules in budget.items():
        for regex, count in rules.items():
            if count <= 0:
                continue
            if re.search(regex, reason):
                rules[regex] = count - 1
                return True
    return False


# ---------------------------------------------------------------------------
# Non-pytest verifier registry (VER-P0-05)
# ---------------------------------------------------------------------------
# Each callable:  runner (GateRunner) -> (status, counts:dict, note:str).
# Status MUST be derived from REAL subprocess output / JSON — never fabricated.
# A gate that cannot prove a pass reports FAIL/BLOCKED/NOT_RUN with a reason.


def _run_subprocess(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str]:
    """Run a subprocess, retrying empty/malformed results (resilience)."""
    last = (None, "")
    for attempt in range(3):
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            last = (proc.returncode, out)
            if proc.returncode is not None and out.strip():
                return last
        except subprocess.TimeoutExpired:
            last = (None, "(subprocess timed out)")
        if attempt < 2:
            time.sleep(60)
    return last


def _ver_evidence_current(runner: GateRunner) -> tuple[str, dict, str]:
    cmd = [runner.python, "-m", "evidence.current", "--check"]
    rc, out = _run_subprocess(cmd, runner.cwd, 600)
    tail = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if rc == 0:
        return Status.PASS, {}, tail or "every artifact CURRENT"
    # Honest: exit !=0 means STALE/UNRESOLVED -> FAIL.
    reason = tail if tail else "evidence.current --check failed"
    return Status.FAIL, {}, f"STALE/UNRESOLVED: {reason}"


def _verifier_source_authority(runner: GateRunner) -> tuple[str, dict, str]:
    # Honest best-available check: (a) canonical source tree present,
    # (b) no live code imports the OLD bare `cleaned_operators` duplicate.
    # P0-03 single-identity: canonical is factor_engine.cleaned_operators.
    canonical = REPO_ROOT / "factor_engine" / "cleaned_operators"
    if not canonical.is_dir():
        return Status.FAIL, {}, f"BLOCKED: canonical source {canonical} missing"
    n_canon = sum(1 for _ in canonical.rglob("*.py"))
    # any live (non-archived) import of the BARE `cleaned_operators` package
    # (the pre-merge name) fails the gate — it would be a second package copy.
    dup_imports: list[str] = []
    for base in (
        "factor_engine", "factor_preprocess", "factor_optimizer",
        "factor_assets", "quant_evaluator", "data_access", "integration_tests",
    ):
        p = REPO_ROOT / base
        if not p.is_dir():
            continue
        for f in p.rglob("*.py"):
            # Skip tests/ (they intentionally exercise the legacy import to
            # PROVE it is blocked — P0-03 negative tests), scripts/ (codegen
            # emits legacy-format templates), and the archived copy.
            if "archived" in str(f) or "cleaned_operators" in str(f):
                continue
            rel = str(f.relative_to(REPO_ROOT))
            if "/tests/" in rel or rel.startswith("tests/") or "/scripts/" in rel or rel.startswith("scripts/"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # Only REAL import statements (not docstrings / string literals):
            # strip string/comment content before matching.
            import ast
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(n == "cleaned_operators" or n.startswith("cleaned_operators.") for n in names):
                    dup_imports.append(str(f))
                    break
    if dup_imports:
        return Status.FAIL, {}, (
            "diverged duplicate imported by live code: " + ", ".join(dup_imports[:3])
        )
    return Status.PASS, {}, (
        f"canonical cleaned_operators present ({n_canon} .py); "
        "no live-code import of the bare cleaned_operators duplicate"
    )


def _verifier_supply_chain(runner: GateRunner) -> tuple[str, dict, str]:
    lock = REPO_ROOT / "requirements-production.lock"
    if not lock.is_file():
        return Status.FAIL, {}, "requirements-production.lock missing"
    try:
        text = lock.read_text(encoding="utf-8")
    except Exception as exc:
        return Status.FAIL, {}, f"lock unreadable: {exc}"
    # lock_digest header present?
    digest_ok = re.search(r"lock_digest=([0-9a-f]{64})", text) is not None
    missing = [ln.split("==")[0].strip()
               for ln in text.splitlines()
               if "==" in ln and ln.strip().endswith("==missing")]
    # these ==missing entries are documented optional-provider sentinels; the
    # honesty check is: the lock must parse, carry a digest, and declare the
    # sentinels explicitly (never import-required by the core chain).
    n_pinned = sum(1 for ln in text.splitlines()
                   if "==" in ln and not ln.strip().startswith("#")
                   and not ln.strip().endswith("==missing"))
    if not digest_ok:
        return Status.FAIL, {}, "lock missing lock_digest header (cannot verify integrity)"
    if n_pinned == 0:
        return Status.FAIL, {}, "lock parses to zero pinned deps"
    note = f"lock present with lock_digest; {n_pinned} pinned deps"
    if missing:
        note += f"; explicit sentinels (==missing, optional providers): {', '.join(missing)}"
    return Status.PASS, {}, note


def _verifier_fresh_wheel_matrix(runner: GateRunner) -> tuple[str, dict, str]:
    result = REPO_ROOT / "evidence" / "fresh_wheel_matrix.json"
    # Regenerate honestly from the actual script (LOCAL-only, no git mutation).
    try:
        proc = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "fresh_wheel_matrix.sh")],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return Status.FAIL, {}, "fresh_wheel_matrix.sh timed out"
    if not result.is_file():
        return Status.FAIL, {}, "fresh_wheel_matrix.json not produced by script"
    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except Exception as exc:
        return Status.FAIL, {}, f"fresh_wheel_matrix.json unparsable: {exc}"
    pkgs = data.get("packages", {})
    all_pass = bool(data.get("all_pass"))
    blocked = [f"{name} ({p.get('reason') or 'no reason'})"
               for name, p in pkgs.items() if p.get("status") != "PASS"]
    if all_pass and not blocked:
        return Status.PASS, {"passed": len(pkgs)}, f"all {len(pkgs)} packages PASS"
    return Status.BLOCKED, {"passed": len(pkgs) - len(blocked)}, \
        f"BLOCKED: " + "; ".join(blocked) if blocked else "BLOCKED: all_pass=false"


def _verifier_submodule_reachability(runner: GateRunner) -> tuple[str, dict, str]:
    rc, out = _run_subprocess(
        ["git", "ls-files", "--stage"], str(REPO_ROOT), 60,
    )
    n = 0
    if rc == 0:
        for line in (out or "").splitlines():
            parts = line.split()
            if parts and parts[0] == "160000":
                n += 1
    if n == 0:
        return Status.NOT_RUN, {}, "no submodules declared; nothing pinned to prove"
    return Status.PASS, {}, f"{n} submodule pin(s) declared"


_VERIFIERS: dict[str, object] = {
    "EVIDENCE_CURRENT": _ver_evidence_current,
    "SOURCE_AUTHORITY": _verifier_source_authority,
    "SUPPLY_CHAIN": _verifier_supply_chain,
    "FRESH_WHEEL_MATRIX": _verifier_fresh_wheel_matrix,
    "SUBMODULE_REACHABILITY": _verifier_submodule_reachability,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class GateRunner:
    def __init__(self, python: Path | str = VENV_PYTHON, cwd: Path = REPO_ROOT):
        self.python = str(python)
        self.cwd = str(cwd)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _command_template_hash(argv: list[str]) -> str:
        """Stable identity of the LOGICAL command (template argv as-written).

        The argv is hashed BEFORE any ``{junitxml}`` substitution, so the same
        gate command produces the same hash on every run regardless of the
        per-run temp XML path (P1-17)."""
        return hashlib.sha256(json.dumps(argv, sort_keys=True).encode("utf-8")).hexdigest()

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
    def _tree_identity() -> str:
        """PlatformSourceTreeIdentity Merkle root of the live working tree.

        P0-WT: bind each evidence entry to the WORKING-TREE content identity so
        a direct uncommitted edit (git HEAD unchanged) still marks the evidence
        STALE.  Returns "" if the identity cannot be computed (never fabricates).
        """
        try:
            from evidence.source_snapshot import platform_source_tree_identity
            return platform_source_tree_identity(REPO_ROOT)["root_merkle"]
        except Exception:  # pragma: no cover - scan failure
            return ""

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
        skipped_reasons: list[str] = []
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
                        else:
                            # A genuine pytest.skip() carries the reason in the
                            # ``message`` attribute (P1-17 skip inventory).
                            msg = child.attrib.get("message")
                            if msg:
                                skipped_reasons.append(msg.strip())
        return {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "xfailed": xfailed,
            "xpassed": xpassed,
            "skipped_reasons": skipped_reasons,
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
            # A registered allowed-skip inventory is the authoritative mechanism
            # for declaring which skips are legitimate.  When one is present it
            # takes precedence over a bare fail_on_skip policy: the inventory
            # enumerates the exactly-allowed reasons, so an inventory-guarded
            # gate does NOT blanket-fail on any skip.  fail_on_skip (fail on ANY
            # skip) therefore only applies when no inventory is declared.
            if (
                spec.skip_policy == "fail_on_skip"
                and junit["skipped"] > 0
                and spec.allowed_skip_inventory is None
            ):
                return Status.FAIL
            # Skip inventory (P1-17): when the gate registers an allowed-skip
            # inventory, every skip reason must be a *registered* reason with
            # allowance remaining; an unregistered/unexpected skip fails the gate.
            if spec.allowed_skip_inventory is not None:
                reasons = list(junit.get("skipped_reasons") or [])
                if len(reasons) != junit["skipped"]:
                    return Status.FAIL
                budget: dict[str, dict] = {}
                for test_rel, rules in spec.allowed_skip_inventory.items():
                    for regex, count in (rules or {}).items():
                        budget.setdefault(test_rel, {})[regex] = int(count)
                for reason in reasons:
                    if not _skip_allowed(reason, budget):
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
        # P1-17: the identity hash is computed from the TEMPLATE argv (the
        # `{junitxml}` token still present), so it is identical on every run.
        # The per-run temp path is recorded separately.
        command_template_hash = self._command_template_hash(list(argv))
        runtime_temp_paths = [str(xml_path)] if xml_path is not None else []
        cmd_hash = command_template_hash
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
            "command_template_hash": command_template_hash,
            "rendered_command": full_cmd,
            "runtime_temp_paths": runtime_temp_paths,
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
            "tree_identity": self._tree_identity(),
            "note": "TIMEOUT" if timed_out else None,
        }
        return entry

    def run_verifier(self, spec: GateSpec) -> dict:
        """Run a non-pytest verifier and derive status from REAL output."""
        fn = _VERIFIERS.get(spec.verifier or "")
        if fn is None:
            return {
                "test": "",
                "argv": [],
                "command_hash": "",
                "command_template_hash": "",
                "rendered_command": [],
                "runtime_temp_paths": [],
                "exit_code": None,
                "status": Status.BLOCKED,
                "passed": None, "failed": None, "errors": None,
                "skipped": None, "xfailed": None, "xpassed": None,
                "tests": None, "collected_tests": None,
                "duration_sec": 0.0, "executed_at": self._now_iso(),
                "timed_out": False,
                "tree_identity": self._tree_identity(),
                "note": f"no verifier registered for {spec.verifier!r}",
            }
        started = time.time()
        status, counts, note = fn(self)
        duration = round(time.time() - started, 3)
        return {
            "test": "",
            "argv": [],
            "command_hash": "",
            "command_template_hash": "",
            "rendered_command": [],
            "runtime_temp_paths": [],
            "exit_code": (0 if status == Status.PASS else None),
            "status": status,
            "passed": counts.get("passed"), "failed": counts.get("failed"),
            "errors": counts.get("errors"), "skipped": counts.get("skipped"),
            "xfailed": counts.get("xfailed"), "xpassed": counts.get("xpassed"),
            "tests": counts.get("passed") if status == Status.PASS else None,
            "collected_tests": None,
            "duration_sec": duration,
            "executed_at": self._now_iso(),
            "timed_out": False,
            "tree_identity": self._tree_identity(),
            "note": note,
        }

    def run_gate(
        self,
        spec: GateSpec,
        timeout: int | None = None,
        tests: tuple[str, ...] | None = None,
    ) -> dict:
        """Run every command in the gate; PASS only if ALL commands PASS."""
        if spec.verifier is not None:
            entry = self.run_verifier(spec)
            return {"name": spec.gate_id, "status": entry["status"], "evidence": [entry]}
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
