#!/bin/bash
# 等修复阶段结束，再跑收尾链（详情页→首页→相关性→注入→COS）。
# 修复失败即中止，不进入收尾（fail-fast，绝不谎报完成）。
set -u
W=/home/sunhaiwei/quant_projects/work/newmining_20260919
LOG=$W/chain2.log
cd /home/sunhaiwei/quant_projects || exit 1

echo "[$(date '+%F %T')] chain2 waiting for repair" >> "$LOG"
while [ "$(cat "$W/repair.status" 2>/dev/null)" = "running" ]; do sleep 60; done
st=$(cat "$W/repair.status" 2>/dev/null)
echo "[$(date '+%F %T')] repair.status=$st" >> "$LOG"
if [ "$st" != "done" ]; then
    echo "[$(date '+%F %T')] repair did not succeed -> aborting, no finalize" >> "$LOG"
    echo "aborted" > "$W/chain2.status"
    exit 1
fi

echo "[$(date '+%F %T')] starting finalize" >> "$LOG"
bash jobs/finalize_newmining.sh >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] finalize rc=$rc status=$(cat "$W/finalize.status" 2>/dev/null)" >> "$LOG"
echo "$([ $rc -eq 0 ] && echo done || echo failed)" > "$W/chain2.status"
exit $rc
