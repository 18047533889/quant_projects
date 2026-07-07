#!/bin/bash
# TASK-ENG-004: block daily/minute cron until main-library pipeline is ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAILY_UPDATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAUSED_FILE="$DAILY_UPDATE_ROOT/CRON_PAUSED"

if [[ -f "$PAUSED_FILE" ]]; then
  echo "❌ TASK-ENG-004: incremental cron is PAUSED."
  echo "   Marker: $PAUSED_FILE"
  sed -n '1,12p' "$PAUSED_FILE"
  exit 1
fi

if [[ "${MASSIVE_CRON_ENABLED:-}" != "1" ]]; then
  echo "❌ TASK-ENG-004: MASSIVE_CRON_ENABLED is not set to 1."
  echo "   Refuse to run production_update_runner / minute scheduler."
  exit 1
fi

exit 0
