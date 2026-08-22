# -*- coding: utf-8
"""config_runtime 与 pipeline 配置贯通测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import load_config
from runtime.config_runtime import resolve_materialize_kwargs, resolve_run_kwargs


def test_resolve_run_kwargs_production_enables_input_dq(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
profile: prod
factor:
  name: x
  expr: rank(close)
data_source:
  type: parquet
  root: /tmp
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    opts = resolve_run_kwargs(loaded)
    assert opts.auto_warmup is True
    assert opts.input_dq_check is True
    assert opts.pit_enforce is True


def test_resolve_materialize_kwargs_staging_target(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
profile: prod
factor:
  name: x
  expr: rank(close)
data_source:
  type: parquet
  root: /tmp
materialization:
  factor_id: prod_factor_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    opts = resolve_materialize_kwargs(loaded)
    assert opts.write_target == "staging_clickhouse"
    assert opts.preserve_invalid_rows is True
    assert opts.dq_check is True
    assert opts.dq_thresholds is not None
    assert opts.dq_thresholds.min_coverage == pytest.approx(0.85)


def test_resolve_materialize_kwargs_incremental_section(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
profile: prod
factor:
  name: x
  expr: rank(close)
data_source:
  type: parquet
  root: /tmp
materialization:
  incremental:
    since: "2024-01-01"
    lookback_extra: 10
    recompute_tail_bars: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    opts = resolve_materialize_kwargs(loaded)
    assert opts.since == "2024-01-01"
    assert opts.lookback_extra == 10
    assert opts.recompute_tail_bars == 3
    assert opts.ch_ensure_table is True


def test_cli_override_beats_config(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
run:
  mode: research
  auto_warmup: false
dq:
  strict: false
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
    loaded = load_config(config)
    opts = resolve_materialize_kwargs(loaded, cli_dq_check=True)
    assert opts.dq_check is True
