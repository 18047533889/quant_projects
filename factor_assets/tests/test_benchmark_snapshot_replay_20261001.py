import pytest

import factor_assets.scripts.benchmark_snapshot_replay_20261001 as replay_benchmark


def test_tiny_snapshot_replay_benchmark_contract():
    result = replay_benchmark._benchmark_scale(asset_count=2, per_asset=2, repetitions=2)
    assert result["assets"] == 2
    assert result["events"] == 4
    assert result["outputs_equal_including_order_and_timestamps"] is True
    assert len(result["trials"]) == 2
    assert result["trials"][0]["route_order"] == ["legacy", "one_pass"]
    assert result["trials"][1]["route_order"] == ["one_pass", "legacy"]
    for trial in result["trials"]:
        assert trial["outputs_equal_including_order_and_timestamps"] is True
        assert trial["legacy_seconds"] >= 0
        assert trial["one_pass_seconds"] >= 0


def test_snapshot_replay_benchmark_rejects_over_budget(monkeypatch):
    monkeypatch.setattr(
        replay_benchmark.sys, "argv",
        ["benchmark", "--assets", "1000", "--events-per-asset", "51"],
    )
    with pytest.raises(SystemExit):
        replay_benchmark._parse_args()


def test_snapshot_replay_benchmark_refuses_receipt_overwrite(tmp_path):
    target = tmp_path / "receipt.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay_benchmark._write_receipt(target, "replacement")
    assert target.read_text(encoding="utf-8") == "original"


def test_snapshot_replay_benchmark_rejects_existing_receipt_early(monkeypatch, tmp_path):
    target = tmp_path / "already.json"
    target.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        replay_benchmark.sys, "argv",
        ["benchmark", "--assets", "2", "--events-per-asset", "1",
         "--repetitions", "1", "--receipt", str(target)],
    )
    monkeypatch.setattr(
        replay_benchmark, "_benchmark_scale",
        lambda *args, **kwargs: pytest.fail("benchmark started before receipt collision check"),
    )
    with pytest.raises(SystemExit, match="already exists"):
        replay_benchmark.main()
    assert target.read_text(encoding="utf-8") == "keep"


def test_snapshot_replay_benchmark_detects_source_change(monkeypatch):
    monkeypatch.setattr(replay_benchmark, "_head", lambda: "new-head")
    monkeypatch.setattr(replay_benchmark, "_source_hashes", lambda: {"source": "changed"})
    with pytest.raises(SystemExit, match="changed during timing"):
        replay_benchmark._ensure_sources_unchanged("old-head", {"source": "old"})
