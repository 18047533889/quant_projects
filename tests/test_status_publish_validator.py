import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "jobs"))

from status_publish_validator import PublishValidationError, validate_publish


CANARY = Path("/home/sunhaiwei/quantsociety/runs/status-canary-20160104-08-r3")


def copy_canary(tmp_path: Path) -> Path:
    workdir = tmp_path / "run"
    shutil.copytree(CANARY, workdir)
    return workdir


def manifest_path(workdir: Path) -> Path:
    return workdir / "manifests" / "feature_manifest.json"


def test_canary_passes_independent_release_gate():
    report = validate_publish(CANARY)
    assert report["status"] == "PASS"
    assert report["output"]["rows"] == 14058
    assert report["quality"]["publishable_rows"] == 12762


@pytest.mark.parametrize("field", ["sha256", "bytes", "rows", "schema_fingerprint"])
def test_output_integrity_mismatch_blocks_publish(tmp_path, field):
    workdir = copy_canary(tmp_path)
    path = manifest_path(workdir)
    data = json.loads(path.read_text())
    if field == "sha256":
        data["output"][field] = "0" * 64
    elif field == "schema_fingerprint":
        data["output"][field] = "1" * 64
    else:
        data["output"][field] += 1
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PublishValidationError, match=f"output\\.{field} mismatch"):
        validate_publish(workdir)


def test_schema_and_required_fields_are_enforced(tmp_path):
    workdir = copy_canary(tmp_path)
    path = manifest_path(workdir)
    data = json.loads(path.read_text())
    del data["quality"]["publishable_rows"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PublishValidationError, match="publishable_rows"):
        validate_publish(workdir)


def test_memory_must_retain_at_least_twenty_percent(tmp_path):
    workdir = copy_canary(tmp_path)
    path = manifest_path(workdir)
    data = json.loads(path.read_text())
    data["memory"]["available_bytes"] = data["memory"]["total_bytes"] // 5 - 1
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PublishValidationError, match="at least 20%"):
        validate_publish(workdir)


def test_pit_flag_and_timestamp_are_checked(tmp_path):
    workdir = copy_canary(tmp_path)
    output = workdir / "outputs" / "part-00000.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(output)
    values = table["pit_validity_flag"].to_pylist()
    values[0] = 0
    table = table.set_column(
        table.column_names.index("pit_validity_flag"),
        "pit_validity_flag",
        pa.array(values, type=table["pit_validity_flag"].type),
    )
    pq.write_table(table, output, compression="zstd", use_dictionary=True)

    with pytest.raises(PublishValidationError, match="PIT violation"):
        validate_publish(workdir)


def test_explicit_future_language_in_manifest_is_blocked(tmp_path):
    workdir = copy_canary(tmp_path)
    path = manifest_path(workdir)
    data = json.loads(path.read_text())
    data["features"][0]["availability_rule"] = "uses future data"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PublishValidationError, match="future/look-ahead"):
        validate_publish(workdir)
