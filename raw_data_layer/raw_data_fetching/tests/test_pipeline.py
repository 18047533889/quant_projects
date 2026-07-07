from __future__ import annotations

from pathlib import Path
import sys

import pytest

_MONOREPO_ROOT = Path(__file__).resolve().parents[3]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))

from raw_data_layer.raw_data_fetching import pipeline

pd = pytest.importorskip("pandas")


def test_validate_parquet_pipeline_returns_summary(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00"]),
            "price": [10.0, 10.5],
            "size": [100.0, 200.0],
        }
    )
    target = tmp_path / "2024-01-02.parquet"
    frame.to_parquet(target, index=False)

    output = pipeline.validate_parquet(str(tmp_path), workers=1)

    assert output["command"] == "validate-parquet"
    assert output["directory"] == str(tmp_path.resolve())
    assert output["total_files"] == 1
    assert output["error_files"] == 0
    assert output["warning_files"] == 0
    assert output["healthy_files"] == 1
    assert output["issue_files"] == []


def test_pipeline_download_all_history_routes_to_runner(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {"command": "download-all-history", "summary_rows": 1}

    monkeypatch.setattr(pipeline, "_load_download_all_history_runner", lambda: fake_runner)

    output = pipeline.download_all_history(api_key="token", root_dir=str(tmp_path), emit_logs=False)

    assert output["command"] == "download-all-history"
    assert calls["api_key"] == "token"
    assert calls["root_dir"] == str(tmp_path)
    assert calls["emit_logs"] is False


def test_pipeline_download_history_routes_to_runner(monkeypatch):
    calls: dict[str, object] = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {"command": "download-history", "mode": "list-keys", "count": 0, "items": []}

    monkeypatch.setattr(pipeline, "_load_download_history_runner", lambda: fake_runner)

    output = pipeline.download_history(
        access_key="ak",
        secret_key="sk",
        prefix="demo/",
        list_keys=1,
        emit_logs=False,
    )

    assert output["command"] == "download-history"
    assert calls["access_key"] == "ak"
    assert calls["secret_key"] == "sk"
    assert calls["prefix"] == "demo/"
    assert calls["list_keys"] == 1
    assert calls["emit_logs"] is False