#!/usr/bin/env bash
# 收尾后补救器：等 finalize_newmining 完成后，再扫一轮 repair-all（残剩因子此时
# 多为 coverage 类，走 eval-only 快速通道），然后重刷全套页面。
set -u
cd /home/sunhaiwei/quant_projects
W=work/newmining_20260919
PY=.venv/bin/python
R=/home/sunhaiwei/quant_project_archives/factor_engine-docs/reports/2026-08-23
L=$W/followup.log
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $L; }

log 'watcher start; waiting for finalize.status=done'
for i in $(seq 1 720); do   # 最长等 12h
    st=$(cat $W/finalize.status 2>/dev/null || echo '')
    [ "$st" = 'done' ] && break
    sleep 60
done
if [ "${st:-}" != 'done' ]; then log 'TIMEOUT waiting finalize; abort (不与链抢资源)'; exit 1; fi
log 'finalize done detected; extra repair-all sweep (2 rounds)'

$PY jobs/new_mining_intake.py --repair-all --repair-rounds 2 --batch-size 8 \
    --threads 6 --root-workers 8 --window-years 3 \
    --memory-gib 64 --host-mem-floor-gib 6 --wave-time-limit-s 21600 >> $L 2>&1
rc=$?; log "repair-all sweep exit=$rc"

log 'republish: pages'
$PY jobs/new_mining_publish.py --pages >> $L 2>&1
log 'republish: index'
$PY jobs/new_mining_publish.py --index >> $L 2>&1
log 'republish: correlate'
$PY jobs/new_mining_publish.py --correlate --ndates 180 >> $L 2>&1
log 'republish: index+corr-inject'
$PY jobs/new_mining_publish.py --index >> $L 2>&1
$PY jobs/new_mining_publish.py --corr-inject >> $L 2>&1
log 'republish: nm_preview'
$PY jobs/nm_preview.py --out $R/nm_preview.html >> $L 2>&1
log 'followup complete'
touch $W/followup.done
