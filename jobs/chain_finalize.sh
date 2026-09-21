#!/bin/bash
# 常驻链：等分片落值全部退出 → 触发收尾流水线 → 记录真实结果。
# 放在服务器上 nohup 运行，不依赖任何本地会话。
# 日志：work/newmining_20260919/chain.log
set -u
cd /home/sunhaiwei/quant_projects || exit 1
W=work/newmining_20260919
LOG=$W/chain.log

echo "[chain] waiting for shard orchestrators at $(date '+%F %T')" >> "$LOG"
while pgrep -f 'new_mining_intake.py --waves' > /dev/null; do
    sleep 60
done
echo "[chain] shards exited at $(date '+%F %T')" >> "$LOG"

# Let in-flight filesystem writes settle before merging shard state.
sleep 20

bash jobs/finalize_newmining.sh
RC=$?

# The exit code is the truth; never claim completion on a failed child.
echo "[chain] finalize rc=$RC status=$(cat "$W/finalize.status" 2>/dev/null || echo unknown) at $(date '+%F %T')" >> "$LOG"
exit "$RC"
