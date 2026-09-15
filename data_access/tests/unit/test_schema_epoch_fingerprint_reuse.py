from types import SimpleNamespace
import pytest
from data_access.read import schema_epoch as mod


def test_manifest_fingerprint_recomputed_from_fields_once_per_epoch(monkeypatch):
    fields={"x":"float64"}
    calls=[]
    real=mod.schema_fingerprint
    def count(schema):
        calls.append(dict(schema))
        return real(schema)
    monkeypatch.setattr(mod,"schema_fingerprint",count)
    def forbidden(path):
        raise AssertionError("covered manifest must not reread")
    monkeypatch.setattr(mod,"parquet_footer_schema",forbidden)
    manifest=SimpleNamespace(epoch_summary_for_paths=lambda paths:(
        {p:"untrusted_claimed_hash" for p in paths},{"untrusted_claimed_hash":fields}))
    gate=mod.SchemaEpochGate(strict=True)
    epochs=gate.group_epochs(["a","b","c"],manifest=manifest)
    assert calls==[fields]
    assert epochs[0].fingerprint==real(fields)
    assert epochs[0].objects==("a","b","c")
    fields["new"]="int64"
    assert gate.group_epochs(["a"],manifest=manifest)[0].fingerprint==real(fields)
    assert len(calls)==2  # no stale cross-call cache


def test_uncovered_footer_hashed_once_and_strict_failure_preserved(monkeypatch):
    real=mod.schema_fingerprint
    calls=[]
    monkeypatch.setattr(mod,"schema_fingerprint",lambda schema:(calls.append(schema) or real(schema)))
    monkeypatch.setattr(mod,"parquet_footer_schema",lambda path:{"x":"float64"})
    result=mod.SchemaEpochGate(strict=True).group_epochs(["a","b"])
    assert len(calls)==2
    assert result[0].objects==("a","b")
    def broken(path):raise OSError("bad footer")
    monkeypatch.setattr(mod,"parquet_footer_schema",broken)
    with pytest.raises(mod.SchemaContractError):
        mod.SchemaEpochGate(strict=True).group_epochs(["broken"])
