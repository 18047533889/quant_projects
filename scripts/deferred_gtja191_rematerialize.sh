#!/usr/bin/env bash
# 延迟任务：factor_engine 算子改完后，由 AI/人工复查再全量落盘。
# 由 schedule_gtja191_rematerialize.sh 在指定延迟后调用。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/logs/deferred_gtja191_rematerialize.log"
FLAG="${ROOT}/.scheduled_gtja191_rematerialize.ready"
STAMP="$(date -Iseconds)"

mkdir -p "${ROOT}/logs"

{
  echo "=== ${STAMP} deferred_gtja191_rematerialize ==="
  echo "触发：请在新 Cursor 对话中让 AI："
  echo "  1) 检查 factor_engine/cleaned_operators 等近期改动"
  echo "  2) 跑 factor_engine/tests 与 verify_gtja191_factor_lake 抽样"
  echo "  3) 确认后再执行全量 force 落盘"
  echo ""
  echo "落盘命令（确认后）："
  echo "  source ${ROOT}/env.sh"
  echo "  unset FACTOR_LAKE_ROOT"
  echo "  python3 ${ROOT}/scripts/materialize_gtja191_factors.py --force --batch-size 1"
  echo ""
} | tee -a "${LOG}"

cat > "${FLAG}" <<EOF
scheduled_at=${STAMP}
action=review_factor_engine_then_materialize
log=${LOG}
command=python3 ${ROOT}/scripts/materialize_gtja191_factors.py --force --batch-size 1
note=用户要求：算子改完后再落盘；勿在未复查前自动 force 全量。
EOF

echo "ready flag -> ${FLAG}" | tee -a "${LOG}"
