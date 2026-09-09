"""Opt-in pytest evidence ledger; a passing unclassified test is not math proof.

Use -p factor_engine.scripts.v8_pytest_evidence --v8-report=/new/report.jsonl.
Explicit categories keep references, parity, rejection and performance separate.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--v8-report", default=None, help="New JSONL test outcome ledger")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "v8_evidence(category, reference_id=None): explicit evidence kind"
    )
    destination = config.getoption("--v8-report")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never append a different source/test run to an old certification ledger.
        config._v8_stream = path.open("x", encoding="utf-8")
        config._v8_phases = {}
        config._v8_completed_nodes = 0


def _write(config, record):
    stream = getattr(config, "_v8_stream", None)
    if stream is None:
        return
    base = dict(nodeid=None, phase=None, outcome=None, category="UNCLASSIFIED",
                wasxfail=None, reference_assertion_passed=False)
    base.update(record)
    stream.write(json.dumps(base, ensure_ascii=False) + "\n")
    stream.flush()


def pytest_collection_finish(session):
    config = session.config
    if not hasattr(config, "_v8_stream"):
        return
    paths = {Path(__file__).resolve()}
    paths.update(Path(item.path).resolve() for item in session.items)
    if config.inipath:
        paths.add(Path(config.inipath).resolve())
    hashes = {}
    for path in sorted(paths):
        try:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            hashes[str(path)] = {"unavailable": type(exc).__name__}
    _write(config, dict(record_type="collection", collected=session.testscollected,
                       source_files=hashes, pytest_version=pytest.__version__,
                       python_version=sys.version, args=list(config.invocation_params.args),
                       identity_scope="collected test modules, evidence plugin and pytest config only; not transitive production source"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    stream = getattr(item.config, "_v8_stream", None)
    if stream is None:
        return
    marker = item.get_closest_marker("v8_evidence")
    category = marker.args[0] if marker and marker.args else "UNCLASSIFIED"
    reference = marker.kwargs.get("reference_id") if marker else None
    wasxfail = getattr(report, "wasxfail", None)
    # Keep setup/call/teardown separately: setup skip is not an executed body,
    # and a successful body with a failed teardown is not a successful node.
    record = {
        "record_type": "phase",
        "nodeid": report.nodeid, "phase": report.when,
        "outcome": report.outcome, "duration_seconds": report.duration,
        "wasxfail": wasxfail,
        "category": category, "reference_id": reference,
        "reference_assertion_passed": False,
        "reason": str(report.longrepr) if report.longrepr else None,
        "boundary": "test outcome only; not automatic operator or production certification",
    }
    _write(item.config, record)
    phases = item.config._v8_phases.setdefault(report.nodeid, {})
    phases[report.when] = record
    if report.when == "teardown":
        passed = (set(phases) == {"setup", "call", "teardown"}
                  and all(p["outcome"] == "passed" and not p["wasxfail"]
                          for p in phases.values()))
        _write(item.config, dict(
            record_type="node_final", nodeid=report.nodeid, phase="node_final",
            outcome="passed" if passed else "not_passed", category=category,
            reference_id=reference,
            reference_assertion_passed=bool(passed and category == "math_reference" and reference),
        ))
        item.config._v8_completed_nodes += 1
        del item.config._v8_phases[report.nodeid]


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if hasattr(config, "_v8_stream"):
        _write(config, dict(
            record_type="session_finish", exitstatus=int(exitstatus),
            collected=session.testscollected, testsfailed=session.testsfailed,
            completed_nodes=config._v8_completed_nodes,
            incomplete_nodes=sorted(config._v8_phases),
            successful=int(exitstatus) == 0 and not config._v8_phases,
            boundary="footer required for completion; earlier node records alone do not certify a complete run",
        ))


def pytest_unconfigure(config):
    stream = getattr(config, "_v8_stream", None)
    if stream is not None:
        stream.close()
