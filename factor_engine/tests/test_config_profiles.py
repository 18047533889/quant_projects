"""Phase 7：配置 profile 合并测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import load_config, load_profile


def test_load_profile_prod_exists():
    payload = load_profile("prod")
    assert "materialization" in payload
    assert "lake_root" in payload["materialization"]
    assert payload["run"]["auto_warmup"] is True
    assert payload["dq"]["profile"] == "us_equity_daily_prod"
    assert payload["materialization"]["target"] == "staging_clickhouse"
    assert payload["materialization"]["preserve_invalid_rows"] is True


def test_load_config_prod_profile_fields(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
profile: prod
factor:
  name: prod_smoke
  expr: rank(close)
data_source:
  type: data_access
  dataset: ashare_stock_daily
  end_date: "2024-02-29"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.run.mode == "production"
    assert loaded.run.auto_warmup is True
    assert loaded.dq.profile == "us_equity_daily_prod"
    assert loaded.dq.strict is True
    assert loaded.materialization is not None
    assert loaded.materialization.target == "staging_clickhouse"
    assert loaded.materialization.preserve_invalid_rows is True


def test_config_merges_profile_overrides(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
profile: dev
factor:
  name: smoke_factor
  expr: rank(close)
data_source:
  type: data_access
  dataset: ashare_stock_daily
  end_date: "2024-02-29"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.materialization is not None
    assert loaded.materialization.lake_root.endswith("lake_dev")
    assert loaded.data_source.options["start_date"] == "2024-01-02"
    assert loaded.data_source.options["end_date"] == "2024-02-29"


def test_unknown_profile_raises(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
profile: does_not_exist
factor:
  name: x
  expr: close
data_source:
  type: parquet
  root: /tmp
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_config(config)
