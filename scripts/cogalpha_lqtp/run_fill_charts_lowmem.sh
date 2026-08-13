#!/usr/bin/env bash
# DISABLED: continuous batch caused OOM on this ~32G/no-swap host.
# Use one-factor-at-a-time instead:
#   bash scripts/cogalpha_lqtp/run_one_weekly_chart.sh
echo "DISABLED: use run_one_weekly_chart.sh (one factor per run)" >&2
exit 1
