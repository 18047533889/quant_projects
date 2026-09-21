#!/bin/bash
# 本周新挖因子：落值完成后的收尾流水线
#   合并分片 → 补跑可修复失败 → 状态 → 详情页 → 首页 → 相关性 → COS 上传
# 每一步都幂等，可安全重复执行。
# 任一步失败即中止，并写 finalize.status=failed；只有全部成功才写 done。
# 绝不谎报完成。日志：work/newmining_20260919/finalize.log
set -u
cd /home/sunhaiwei/quant_projects || exit 1
W=work/newmining_20260919
LOG=$W/finalize.log
STATUS=$W/finalize.status
PY=.venv/bin/python

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

step() {
    local label="$1"; shift
    log "$label"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        log "!!! FAILED (rc=$rc): $label"
        echo "failed" > "$STATUS"
        log "=== finalize aborted ==="
        exit "$rc"
    fi
}

echo "running" > "$STATUS"
log "=== finalize start (pid $$) ==="

step "[1/7] merge shards" \
    $PY jobs/new_mining_shards.py --merge --shards 2

# Retry only failures whose cause is environmental (admission denial caused by a
# concurrent shard) or needs a different execution backend, plus candidates that
# have no matrix on disk at all.  A warm-up shortfall is *not* retried here: the
# panel starts on the report window start, so no earlier history exists — that
# class is fixed by the coverage rule and handled as an eval-only re-check.
# Must run with no other FactorEngine job on the host.  Waves stay at 8 factors
# so a kill costs one wave's landing, but the root-worker pool is widened (8) and
# the per-process ceiling raised (40GiB) to cut the serial landing time.  A kill
# now only costs the landing: each factor is scored and published the moment it
# lands, so nothing already evaluated is lost with the wave.
step "[2/7] repair retryable failures (serial; requires an idle FactorEngine host)" \
    $PY jobs/new_mining_intake.py --repair-all --repair-rounds 3 --batch-size 8 \
        --threads 6 --root-workers 8 --window-years 3 \
        --memory-gib 30 --host-mem-floor-gib 6 --wave-time-limit-s 21600

step "[3/7] status" \
    $PY jobs/new_mining_intake.py --status

step "[4/7] detail pages" \
    $PY jobs/new_mining_publish.py --pages

step "[5/7] homepage new-mining section + master table" \
    $PY jobs/new_mining_publish.py --index

step "[6/7] correlation analysis" \
    $PY jobs/new_mining_publish.py --correlate --ndates 180

# The demotion of pool-redundant / twin factors needs the correlation results,
# so the homepage section is rebuilt *after* correlate, then the correlation
# section is injected (idempotent).
step "[6b/7] rebuild homepage new-mining section with redundancy demotions" \
    $PY jobs/new_mining_publish.py --index

step "[6c/7] inject correlation section" \
    $PY jobs/new_mining_publish.py --corr-inject

step "[7/7] COS upload (values + evaluations + report), readback verified" \
    $PY jobs/new_mining_publish.py --cos

echo "done" > "$STATUS"
log "=== finalize done ==="
