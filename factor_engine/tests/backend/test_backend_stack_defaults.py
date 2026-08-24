# -*- coding: utf-8
"""默认 backend 与底层栈（DuckDB / auto / duckdb_sql）约定测试。"""

from __future__ import annotations

from pathlib import Path

import yaml

from factor_engine.runtime.config import load_config, load_profile

ROOT = Path(__file__).resolve().parents[2]


def test_yaml_default_backend_is_auto(tmp_path):
    cfg = tmp_path / "minimal.yaml"
    cfg.write_text(
        """
factor:
  name: f1
  expr: close
data_source:
  type: data_access
  dataset: us_stocks_sip_day_aggs
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.backend.type == "auto"


def test_prod_profile_recommends_auto_backend():
    payload = load_profile("prod")
    assert payload["backend"]["type"] == "auto"


def test_data_access_auto_smoke_yaml_loads():
    path = ROOT / "examples" / "configs" / "data_access_auto_smoke.yaml"
    loaded = load_config(path)
    assert loaded.data_source.type == "data_access"
    assert loaded.backend.type == "auto"


def test_data_access_duckdb_sql_smoke_yaml_loads():
    path = ROOT / "examples" / "configs" / "data_access_duckdb_sql_smoke.yaml"
    loaded = load_config(path)
    assert loaded.data_source.type == "data_access"
    assert loaded.backend.type == "duckdb_sql"


def test_data_access_template_uses_auto():
    path = ROOT / "examples" / "config_us_stock_pv_data_access_template.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["backend"]["type"] == "auto"


def test_backend_config_dataclass_default_is_auto():
    from factor_engine.runtime.config import BackendConfig

    assert BackendConfig().type == "auto"
