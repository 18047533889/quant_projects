import pytest
import json

from quant_evaluator.scripts import preflight_materialized_f48_auto as preflight
from quant_evaluator.scripts import load_real_cos_factor_batch as loader


def _records(count=48, size=1024):
    records = {}
    for index in range(count):
        factor_id = f"f{index:02d}"
        digest = f"{index + 1:064x}"
        records[factor_id] = {
            "verified": True,
            "status": "materialized_not_evaluated",
            "bytes": size,
            "sha256": digest,
            "uri": f"{preflight.BASE}/factor_values/{digest}/{factor_id}.parquet",
        }
    return records


def test_manifest_selector_requires_48_unique_hashes_and_exact_uris():
    selected, total = preflight.select_records(_records())
    assert len(selected) == 48
    assert len({row["sha256"] for _, row in selected}) == 48
    assert total == 48 * 1024
    records = _records()
    records["f47"]["sha256"] = records["f46"]["sha256"]
    records["f47"]["uri"] = (
        f"{preflight.BASE}/factor_values/{records['f47']['sha256']}/f47.parquet"
    )
    with pytest.raises(ValueError, match="hashes are not distinct"):
        preflight.select_records(records)


def test_manifest_selector_enforces_per_object_and_total_caps():
    records = _records(size=preflight.MAX_OBJECT_MIB * 1024**2 + 1)
    with pytest.raises(ValueError, match="fewer than 48"):
        preflight.select_records(records)
    records = _records(size=preflight.MAX_TOTAL_MIB * 1024**2 // 48 + 1)
    with pytest.raises(ValueError, match="exceed 2048 MiB"):
        preflight.select_records(records)


def test_loader_explicit_f48_bounds_validate_before_any_io(monkeypatch):
    class ReachedIO(Exception):
        pass
    monkeypatch.setattr(loader, "DuckDBEngine",
                        lambda **kwargs: (_ for _ in ()).throw(ReachedIO()))
    with pytest.raises(ReachedIO):
        list(loader._iter_factors(
            48, 128, preflight.MANIFEST_SHA256, 2048))
    with pytest.raises(ValueError, match="factor count 1..48"):
        list(loader._iter_factors(
            49, 128, preflight.MANIFEST_SHA256, 2048))
    with pytest.raises(ValueError, match="1..2048 MiB"):
        list(loader._iter_factors(
            48, 128, preflight.MANIFEST_SHA256, 2049))


def test_preflight_failure_is_json_and_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "preflight", lambda: {
        "status": "insufficient_host_memory", "pass": False,
        "mem_available_bytes": 1,
    })
    assert preflight.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "insufficient_host_memory"
    assert report["pass"] is False
