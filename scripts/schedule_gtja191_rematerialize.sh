#!/usr/bin/env bash
# 安排延迟执行 deferred_gtja191_rematerialize（默认 1 小时）。
# 用法:
#   bash scripts/schedule_gtja191_rematerialize.sh          # 1 小时后
#   bash scripts/schedule_gtja191_rematerialize.sh 30m     # 30 分钟后
#   bash scripts/schedule_gtja191_rematerialize.sh cancel  # 取消已排任务
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DELAY="${1:-1h}"
PIDFILE="${ROOT}/logs/schedule_gtja191_rematerialize.pid"
LOG="${ROOT}/logs/schedule_gtja191_rematerialize.log"

cancel() {
  if [[ -f "${PIDFILE}" ]]; then
    pid="$(cat "${PIDFILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      echo "已取消延迟任务 pid=${pid}"
    fi
    rm -f "${PIDFILE}"
  else
    echo "无排期任务"
  fi
}

if [[ "${DELAY}" == "cancel" ]]; then
  cancel
  exit 0
fi

cancel

# 解析延迟：1h / 30m / 3600
case "${DELAY}" in
  *h) SECS=$(( ${DELAY%h} * 3600 )) ;;
  *m) SECS=$(( ${DELAY%m} * 60 )) ;;
  *)  SECS="${DELAY}" ;;
esac

RUN_AT="$(date -d "+${SECS} seconds" -Iseconds 2>/dev/null || date -Iseconds)"

nohup bash -c "
  sleep ${SECS}
  bash '${ROOT}/scripts/deferred_gtja191_rematerialize.sh'
" >> "${LOG}" 2>&1 &

echo $! > "${PIDFILE}"
echo "已排期：${SECS}s 后（约 ${RUN_AT}）执行 deferred_gtja191_rematerialize"
echo "pid=$(cat "${PIDFILE}") log=${LOG}"
echo "取消：bash scripts/schedule_gtja191_rematerialize.sh cancel"
