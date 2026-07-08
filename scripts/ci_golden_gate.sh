#!/usr/bin/env bash
# Phase 7 双市场 golden CI 门禁：契约 + golden 回归（无外部 parquet 依赖）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== data_access market contracts =="
cd "${ROOT}/data_access"
PYTHONPATH=. python3 -m pytest \
  tests/contract/test_store_market_datasets.py \
  tests/contract/test_us_instrument_column_contract.py \
  -q

echo "== factor_engine golden regression =="
cd "${ROOT}/factor_engine"
PYTHONPATH=.:"${ROOT}" python3 -m pytest \
  tests/test_golden_regression.py \
  tests/test_golden_market_regression.py \
  tests/test_config_profiles.py \
  tests/test_phase7_ops.py \
  tests/test_phase8_ops.py \
  tests/test_phase9_ops.py \
  -q

echo "golden gate: OK"
