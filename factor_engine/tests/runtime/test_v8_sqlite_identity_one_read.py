import hashlib
import json
import os
import uuid
from pathlib import Path

import factor_engine.runtime.worker_ownership_sqlite as store
from factor_engine.runtime.worker_ownership import RunOwnershipContext


def _context(run_dir):
    run_id = uuid.uuid4().hex
    payload = json.dumps(
        {"run_id": run_id, "ownership_store": "sqlite-v1"},
        separators=(",", ":"),
    ).encode()
    (run_dir / "identity.json").write_bytes(payload)
    return RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


def test_sqlite_identity_is_authenticated_once_without_path_reopen(tmp_path, monkeypatch):
    context = _context(tmp_path)
    identity_path = tmp_path / "identity.json"
    real_open = store.os.open
    identity_opens = []

    def tracked_open(path, flags, *args, **kwargs):
        if Path(path) == identity_path:
            identity_opens.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def forbidden_read_bytes(_path):
        raise AssertionError("identity path was reopened after authentication")

    monkeypatch.setattr(store.os, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    store.initialize_worker_store(tmp_path, context=context)

    assert len(identity_opens) == 1
    assert identity_opens[0] & getattr(os, "O_NOFOLLOW", 0) == getattr(
        os, "O_NOFOLLOW", 0
    )
    assert identity_opens[0] & getattr(os, "O_NONBLOCK", 0) == getattr(
        os, "O_NONBLOCK", 0
    )
    assert (tmp_path / store.STORE_NAME).is_file()


def test_sqlite_selection_uses_exact_mapping_returned_by_verifier(tmp_path, monkeypatch):
    context = _context(tmp_path)
    calls = []

    def authenticated(_run_dir, checked_context):
        calls.append(checked_context)
        return {"run_id": context.run_id, "ownership_store": "sqlite-v1"}

    monkeypatch.setattr(store, "_verify_context", authenticated)
    monkeypatch.setattr(
        Path, "read_bytes",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("untrusted identity reread")
        ),
    )

    store._verify_sqlite_identity(tmp_path, context)

    assert calls == [context]
