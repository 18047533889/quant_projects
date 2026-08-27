#!/usr/bin/env bash
# Phase 5 P1-16：统一 FactorEngine 验收入口（make audit-factor-engine 调它）。
#
# 至少覆盖：FE 全量 + DA 全量 + 资源治理单测 + 并发/缓存失效 + allowlist。
# 失败即非零退出。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FE="$ROOT/factor_engine"
DA="$ROOT/data_access"
FAIL=0

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[1;31mFAIL: %s\033[0m\n' "$*"; FAIL=1; }
pass() { printf '\033[1;32mok: %s\033[0m\n' "$*"; }

step "1/5 FactorEngine 资源治理单测"
if (cd "$FE" && timeout 600 python3 -m pytest tests/unit/test_resource_governance.py -q -p no:cacheprovider >/tmp/fe_rg.log 2>&1); then
    pass "resource_governance"; else fail "resource_governance (见 /tmp/fe_rg.log)"; fi

step "2/5 FactorEngine 全量（忽略既有 evidence 失败）"
if (cd "$FE" && timeout 1800 python3 -m pytest tests/ -q -p no:cacheprovider >/tmp/fe_full.log 2>&1); then
    pass "factor_engine full"; else fail "factor_engine full (见 /tmp/fe_full.log 尾部)"; fi

step "3/5 DataAccess 全量"
if (cd "$DA" && timeout 1800 python3 -m pytest tests/ -q -p no:cacheprovider >/tmp/da_full.log 2>&1); then
    pass "data_access full"; else fail "data_access full (见 /tmp/da_full.log 尾部)"; fi

step "4/5 allowlist 静态检查"
if (cd "$ROOT" && timeout 600 python3 -m pytest "$DA/tests/unit/test_allowlist_imports.py" -q -p no:cacheprovider >/tmp/da_allowlist.log 2>&1); then
    pass "allowlist"; else fail "allowlist (见 /tmp/da_allowlist.log 尾部)"; fi

step "5/5 语义契约审计"
if (cd "$DA" && timeout 600 python3 -m pytest tests/unit/test_semantic_consistency.py -q -p no:cacheprovider >/tmp/da_semantic.log 2>&1); then
    pass "semantic"; else fail "semantic (见 /tmp/da_semantic.log 尾部)"; fi

if [ "$FAIL" -ne 0 ]; then
    printf '\n\033[1;31m audit-factor-engine: %d 项失败\033[0m\n' "$FAIL"
    exit 1
fi
printf '\n\033[1;32m audit-factor-engine: 全部通过\033[0m\n'
