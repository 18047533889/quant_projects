"""Pipeline 重试与结果 JSON 字段测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

from pipeline import _execute_config_with_retries, run_from_config
from runtime.config import FactorEngineConfig


def _minimal_config() -> FactorEngineConfig:
    from runtime.config import (
        BackendConfig,
        DataSourceConfig,
        EngineConfig,
        FactorDefinitionConfig,
    )

    return FactorEngineConfig(
        factor=FactorDefinitionConfig(name="x", expr='col("close")'),
        data_source=DataSourceConfig(type="memory", options={}),
        backend=BackendConfig(type="pandas"),
        engine=EngineConfig(enable_cache=False),
    )


def test_execute_config_with_retries_recovers_on_transient_failure():
    calls = {"n": 0}

    def flaky_execute(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"analysis": None, "plan": None, "result": pd.Series(dtype=float)}

    with patch("pipeline._execute_config", side_effect=flaky_execute):
        out = _execute_config_with_retries(
            _minimal_config(),
            config_name="x",
            materialize=False,
            preview_rows=0,
            max_retries=2,
        )
    assert calls["n"] == 2
    assert "result" in out


def test_execute_config_with_retries_raises_after_exhausted():
    with patch("pipeline._execute_config", side_effect=RuntimeError("permanent")):
        with pytest.raises(RuntimeError, match="permanent"):
            _execute_config_with_retries(
                _minimal_config(),
                config_name="x",
                materialize=False,
                preview_rows=0,
                max_retries=1,
            )


def test_pipeline_result_json_includes_input_dq(tmp_path: Path):
    data_root = tmp_path / "day_aggs_v1" / "2024" / "01"
    data_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "window_start": [pd.Timestamp("2024-01-02", tz="UTC").value],
            "close": [10.0],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "volume": [1000.0],
            "transactions": [10.0],
        }
    )
    frame.to_parquet(data_root / "2024-01-02.parquet", index=False)

    config_path = tmp_path / "dq_factor.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "factor": {"name": "close_only", "expr": 'col("close")'},
                "data_source": {
                    "type": "parquet_kline",
                    "root": str(tmp_path / "day_aggs_v1"),
                    "instrument_column": "ticker",
                    "timestamp_column": "window_start",
                    "fields": {"close": "close"},
                },
                "backend": {"type": "pandas"},
                "engine": {"enable_cache": False},
            }
        ),
        encoding="utf-8",
    )

    out = run_from_config(
        config_path,
        output_root=tmp_path / "out",
        input_dq_check=True,
    )
    assert out["results"][0]["status"] == "success"
    assert out["results"][0]["input_dq"]["passed"] is True

    result_json = tmp_path / "out" / "results" / "dq_factor.json"
    assert result_json.exists()
    import json

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["input_dq"]["passed"] is True
