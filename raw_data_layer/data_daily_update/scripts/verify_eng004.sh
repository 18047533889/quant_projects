#!/bin/bash
# TASK-ENG-004 acceptance: cron must stay paused; schedulers must fail fast.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

fail=0

echo "=== TASK-ENG-004 verify ==="

if [[ -f "../CRON_PAUSED" ]]; then
  echo "OK  CRON_PAUSED marker present"
else
  echo "FAIL  CRON_PAUSED marker missing"
  fail=1
fi

if crontab -l 2>/dev/null | grep -qE 'daily_update_scheduler|minute_update_scheduler|production_update_runner'; then
  echo "FAIL  crontab still contains Massive incremental jobs:"
  crontab -l | grep -E 'daily_update_scheduler|minute_update_scheduler|production_update_runner' || true
  fail=1
else
  echo "OK  crontab has no Massive incremental jobs"
fi

if bash daily_update_scheduler.sh >/tmp/eng004_daily.out 2>&1; then
  echo "FAIL  daily_update_scheduler.sh should exit non-zero"
  fail=1
else
  echo "OK  daily_update_scheduler.sh blocked"
fi

if bash minute_update_scheduler.sh >/tmp/eng004_minute.out 2>&1; then
  echo "FAIL  minute_update_scheduler.sh should exit non-zero"
  fail=1
else
  echo "OK  minute_update_scheduler.sh blocked"
fi

if MASSIVE_ALLOW_CRON_INSTALL=1 bash -c 'echo 5 | bash install_cron_jobs.sh' >/tmp/eng004_install.out 2>&1; then
  echo "FAIL  install_cron_jobs.sh should refuse even with ALLOW flag during PAUSED"
  fail=1
else
  echo "OK  install_cron_jobs.sh refuses install while PAUSED"
fi

if [[ "$fail" -eq 0 ]]; then
  echo "PASS  TASK-ENG-004"
  exit 0
fi

echo "FAIL  TASK-ENG-004 — see messages above"
exit 1
