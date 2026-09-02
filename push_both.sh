#!/bin/bash
# push_both.sh — sync 13 library directories from quant_projects to their
# HKUST-QUANT-SOCIETY mirrors on GitHub, then push the root repo itself to
# github.com/18047533889/quant_projects (origin main).
#
# Usage:
#   bash push_both.sh                 # sync all 13 repos + push root repo
#   bash push_both.sh --only fe       # sync repos whose name contains "fe"
#                                     # (--only skips the root-repo push)
#
# Design notes:
#   * Code is synced whole (never split); only regenerable artifacts are excluded
#     (__pycache__, npy caches, backup dirs, vectorbt examples/output, ...).
#   * Tokens are NOT stored in this file. They are read from git remote URLs:
#       origin    -> github.com/18047533889/quant_projects      (user token)
#       hkust-org -> github.com/HKUST-QUANT-SOCIETY/...         (org credential anchor)
#   * Mirror clones are cached in ~/.cache/qs_sync so the 1.1G factor_engine
#     remote is cloned once, not on every sync.
#   * On first sync each HKUST repo also gets a `dev` snapshot branch (== main).

set -uo pipefail

ROOT="/home/sunhaiwei/quant_projects"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/qs_sync"
ORG="HKUST-QUANT-SOCIETY"
LOG="$ROOT/logs/push_both.log"
mkdir -p "$CACHE" "$(dirname "$LOG")"

# ---- credentials: derived from existing remotes, no secrets here ----
hkust_url=$(git -C "$ROOT" remote get-url hkust-org 2>/dev/null)
if [ -z "$hkust_url" ]; then
    echo "FATAL: remote 'hkust-org' missing (credential anchor for HKUST pushes)" >&2
    exit 1
fi
CRED=$(sed -E 's#https?://([^@/]+)@.*#\1#' <<<"$hkust_url")        # e.g. gho_xxx
HOST=$(sed -E 's#https?://[^@/]+@([^/]+)/.*#\1#' <<<"$hkust_url")  # e.g. github.com
org_url() { echo "https://${CRED}@${HOST}/${ORG}/$1.git"; }

GIT_ID=(-c user.name="Sun Haiwei" -c user.email="sunhaiwei@users.noreply.github.com")

# ---- repo table: local dir == HKUST repo name ----
REPOS=(
    factor_engine data_access vectorbt_qs riskfolio_qs
    quant_evaluator quant_platform modeling factor_preprocess
    factor_optimizer factor_assets alphaprobe platform_web lightgbm_qs
)

# ---- excludes: junk & regenerable artifacts only, never source code ----
EXCLUDES=(
    --exclude=.git/ --exclude=__pycache__/ --exclude='*.py[cod]'
    --exclude=.pytest_cache/ --exclude=.mypy_cache/ --exclude=.ruff_cache/
    --exclude=.ipynb_checkpoints/ --exclude='*.log' --exclude=.DS_Store
    --exclude=.venv/ --exclude=venv/ --exclude='*.egg-info/' --exclude=build/
    --exclude=examples/output/                                # vectorbt backtest artifacts
    --exclude='docs/reports/*/factors/'                       # regenerable HTML (jobs/rebuild_factor_detail_pages.py)
    --exclude='docs/reports/*/factors_UNADJ_BACKUP_*/'
    --exclude='docs/reports/*/robustness_2026/r26_cache/'     # regenerable robustness npy cache
    --exclude='docs/reports/*/backup_*/'
    --exclude='*.tar.gz'
    --exclude='data/'                                       # local data (mirror-root relative; source .gitignore ignores it too)
    --exclude='outputs/'                                    # regenerable backtest artifacts
    --exclude='snapshots/'                                  # regenerable training snapshots
)

sync_repo() {
    local src="$ROOT/$1" name="$1" work="$CACHE/$1"
    local url; url=$(org_url "$name")
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$ts] sync $name -> $ORG/$name ===" | tee -a "$LOG"

    # one-time setup: clone existing remote, or init for a brand-new repo
    if [ ! -d "$work/.git" ]; then
        if [ -n "$(git ls-remote "$url" refs/heads/main 2>/dev/null)" ]; then
            git clone -q "$url" "$work" \
                || { echo "    clone FAILED" | tee -a "$LOG"; return 1; }
        else
            git init -q -b main "$work" \
                && git -C "$work" remote add origin "$url" \
                || { echo "    init FAILED" | tee -a "$LOG"; return 1; }
        fi
    fi

    # align mirror working tree with remote main (no-op for empty/new repos)
    if git -C "$work" fetch -q origin main 2>>"$LOG"; then
        if git -C "$work" rev-parse -q --verify FETCH_HEAD >/dev/null 2>&1; then
            git -C "$work" checkout -q -B main FETCH_HEAD
        fi
    fi

    rsync -a --delete "${EXCLUDES[@]}" "$src/" "$work/" 2>>"$LOG" \
        || { echo "    rsync FAILED" | tee -a "$LOG"; return 1; }

    git -C "$work" add -A
    if git -C "$work" diff --cached --quiet; then
        echo "    no changes" | tee -a "$LOG"
        return 0
    fi

    git -C "$work" "${GIT_ID[@]}" commit -q -m "sync from quant_projects: $ts" \
        || { echo "    commit FAILED" | tee -a "$LOG"; return 1; }
    git -C "$work" push -q origin main \
        || { echo "    push FAILED" | tee -a "$LOG"; return 1; }

    # keep a dev snapshot branch on first sync (never force-moved afterwards)
    if [ -z "$(git -C "$work" ls-remote --heads origin dev 2>/dev/null)" ]; then
        git -C "$work" push -q origin main:dev 2>>"$LOG"
    fi
    echo "    done (pushed main)" | tee -a "$LOG"
}

ONLY=""
if [ "${1:-}" = "--only" ]; then ONLY="${2:-}"; fi

fail=0
for r in "${REPOS[@]}"; do
    if [ -n "$ONLY" ] && [[ "$r" != *"$ONLY"* ]]; then continue; fi
    sync_repo "$r" || fail=1
done

if [ "$fail" -eq 0 ]; then
    echo "=== all syncs complete ===" | tee -a "$LOG"
else
    echo "=== syncs finished WITH ERRORS (see $LOG) ===" | tee -a "$LOG"
fi

# ---- push the root repo itself to origin (18047533889/quant_projects) ----
# --only 模式跳过总仓推送（只刷个别库时不动 root 历史）
# push 只发送已提交历史，脏树不影响正确性 —— 脏文件仅提示不阻断
if [ -z "$ONLY" ] && [ "$fail" -eq 0 ]; then
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$ts] push root repo -> 18047533889/quant_projects ===" | tee -a "$LOG"
    if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
        echo "    WARN: root working tree dirty ($(git -C "$ROOT" status --porcelain | wc -l) files uncommitted)" | tee -a "$LOG"
    fi
    git -C "$ROOT" push -q origin main 2>>"$LOG" \
        && echo "    done (pushed root main)" | tee -a "$LOG" \
        || { echo "    root push FAILED" | tee -a "$LOG"; fail=1; }
fi

exit "$fail"
