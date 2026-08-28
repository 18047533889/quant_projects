#!/bin/bash
# Sync directories to HKUST org repos after push to quant_projects

REMOTE="$1"
BRANCH="$2"

if [ "$REMOTE" = "origin" ] && [ "$BRANCH" = "refs/heads/main" ]; then
    TEMP=$(mktemp -d)
    TOKEN="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"
    
    sync_dir() {
        local SOURCE="$1"
        local HKUST_REPO="$2"
        local NAME=$(basename "$SOURCE")
        
        echo "=== Syncing $NAME to HKUST ==="
        
        git clone "https://${TOKEN}@${HKUST_REPO}" "$TEMP/$NAME"
        
        # Remove all files except .git
        rm -rf "$TEMP/$NAME"/*
        
        # Copy source to temp (exclude .git)
        rsync -av --exclude='.git' "$SOURCE/" "$TEMP/$NAME/"
        
        # Commit and push
        (
            cd "$TEMP/$NAME"
            git add -A
            git commit -m "Sync from quant_projects: $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || echo "No changes"
            git push "https://${TOKEN}@${HKUST_REPO}" main
        )
        
        rm -rf "$TEMP/$NAME"
        echo "=== $NAME sync done ==="
    }
    
    sync_dir "/home/sunhaiwei/quant_projects/factor_engine" "github.com/HKUST-QUANT-SOCIETY/factor_engine.git"
    sync_dir "/home/sunhaiwei/quant_projects/vectorbt_qs" "github.com/HKUST-QUANT-SOCIETY/vectorbt_qs.git"
    sync_dir "/home/sunhaiwei/quant_projects/data_access" "github.com/HKUST-QUANT-SOCIETY/data_access.git"
    
    rm -rf "$TEMP"
    echo "=== All syncs complete ==="
fi
