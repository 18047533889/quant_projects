import subprocess

import pytest

from quant_evaluator.scripts import profile_real_cos_single_object as profiler


def test_instrumentation_delegates_and_restores_after_exception(monkeypatch):
    import data_access.cos.remote as remote
    import data_access.cos.research as research
    from data_access.read.read_handle import ReadHandle
    from data_access.store import DataAccessStore

    uri = "cos://bucket/factor.parquet"
    calls = []

    def fake_head(request_uri, *args, **kwargs):
        calls.append(("head", request_uri))
        return {"etag": "etag", "size": 1}

    def fake_run(argv, *args, **kwargs):
        calls.append(("run", tuple(argv)))
        return "delegated"

    monkeypatch.setattr(remote, "cos_cli_head", fake_head)
    monkeypatch.setattr(subprocess, "run", fake_run)
    research_subprocess = research.subprocess
    research_hashlib = research.hashlib
    path_open = profiler.Path.open
    arrow = ReadHandle.to_arrow
    store_read = DataAccessStore.read
    ledger = profiler.PhaseLedger()

    with pytest.raises(RuntimeError, match="exercise cleanup"):
        with profiler.instrument_existing_read(
                ledger, exact_uri=uri, cli="/approved/coscli",
                factor_dataset="factor_panel"):
            remote.cos_cli_head("cos://other/ignored.parquet")
            remote.cos_cli_head(uri)
            assert research.subprocess.run(["other", "args"]) == "delegated"
            assert research.subprocess.run(
                ["/approved/coscli", "cp", uri, "/private/object.parquet"]) == "delegated"
            raise RuntimeError("exercise cleanup")

    assert calls == [
        ("head", "cos://other/ignored.parquet"),
        ("head", uri),
        ("run", ("other", "args")),
        ("run", ("/approved/coscli", "cp", uri, "/private/object.parquet")),
    ]
    assert ledger.counts == {"head_s": 1, "download_s": 1}
    assert remote.cos_cli_head is fake_head
    assert research.subprocess is research_subprocess
    assert research.hashlib is research_hashlib
    assert profiler.Path.open is path_open
    assert ReadHandle.to_arrow is arrow
    assert DataAccessStore.read is store_read
