"""Versioned publication of freshly executed native parameter point evidence.

This does not grant operator lifecycle status or execution receipts. Those
independent gates must also admit the actual selected implementation.
"""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path


def capture_generation(canonicals):
    from factor_engine.scripts.audit_r37_parameter_domains import _require_clean_worktree
    from factor_engine.backend.native_polars_evidence import _identity
    from factor_engine.runtime.parameter_domain_store import FE_ROOT
    names = tuple(sorted(set(canonicals)))
    if not names:
        raise ValueError("at least one canonical is required")
    _, head, hashes = _require_clean_worktree(cwd=FE_ROOT)
    identities = {name: _identity(name) for name in names}
    paths = [FE_ROOT / "scripts/native_polars_parameter_points.py",
             FE_ROOT / "scripts/native_polars_parameter_publication.py",
             FE_ROOT / "tests/scripts/test_native_polars_parameter_points.py",
             FE_ROOT / "tests/scripts/test_native_polars_parameter_publication.py",
             FE_ROOT / "backend/native_polars_evidence.py",
             FE_ROOT / "backend/panel_polars.py",
             FE_ROOT / "scripts/native_receipt_publication.py"]
    for identity in identities.values():
        implementation = Path(identity["implementation_file"]).resolve()
        implementation.relative_to(FE_ROOT.resolve())
        # Include same-module helper inputs, not only the selected class body.
        paths.extend(implementation.parent.glob("*.py"))
    for path in paths:
        relative = str(path.resolve().relative_to(FE_ROOT.resolve()))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"canonicals": names, "head": head, "hashes": hashes,
            "identities": identities}


def publish_store(store, output, generation):
    """Refuse stale tests, source drift, empty evidence, and overwrites."""
    from factor_engine.runtime.parameter_domain_store import FE_ROOT
    from factor_engine.scripts.native_receipt_publication import publish_receipts
    output = Path(output).resolve()
    approved = (FE_ROOT.parent / "evidence").resolve()
    try:
        output.relative_to(approved)
    except ValueError as exc:
        raise ValueError("native parameter evidence must be inside project evidence") from exc
    if output.exists():
        raise FileExistsError("use a fresh versioned parameter evidence path")
    if capture_generation(generation["canonicals"]) != generation:
        raise RuntimeError("native parameter source generation changed during tests")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,
                                         prefix=".parameter-points-", suffix=".json",
                                         delete=False) as handle:
            pending = Path(handle.name)
        store.save_json(pending)
        payload = json.loads(pending.read_text())
        points = payload.get("certified_points", [])
        if not points or any(row.get("backend") != "polars"
                             or row.get("source") != "fresh_selected_polars_authority_panel_parity"
                             or not row.get("evidence_hash") for row in points):
            raise ValueError("only freshly executed native Polars point evidence may publish")
        allowed = set(generation["canonicals"])
        if any(row.get("canonical") not in allowed for row in points):
            raise ValueError("parameter evidence contains an uncaptured canonical")
        if any(row.get("details", {}).get("native_identity")
               != generation["identities"][row["canonical"]] for row in points):
            raise ValueError("parameter evidence belongs to a stale native implementation")
        hashes = generation["hashes"]
        manifest = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
        payload["_meta"].update({
            "generated_commit": generation["head"], "source_tree": "working_tree",
            "source_input_hashes": hashes,
            "source_input_manifest_hash": hashlib.sha256(manifest.encode()).hexdigest(),
        })
        if capture_generation(generation["canonicals"]) != generation:
            raise RuntimeError("native parameter source generation changed before publication")
        publish_receipts(output, payload)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
