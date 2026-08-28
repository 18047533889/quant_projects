#!/bin/bash
# Push to both repos: quant_projects (full) and HKUST/factor_engine (factor_engine only)

# Get the remote being pushed to
REMOTE="$1"
BRANCH="$2"

# Only sync when pushing to origin main
if [ "$REMOTE" = "origin" ] && [ "$BRANCH" = "refs/heads/main" ]; then
    echo "=== Syncing factor_engine to HKUST repo ==="
    
    # Create temp directory
    TEMP=$(mktemp -d)
    HKUST_TOKEN="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"
    SOURCE="/home/sunhaiwei/quant_projects/factor_engine"
    
    # Clone HKUST repo (will have .git inside)
    git clone "https://${HKUST_TOKEN}@github.com/HKUST-QUANT-SOCIETY/factor_engine.git" "$TEMP"
    
    # Remove all files except .git
    rm -rf "$TEMP"/*
    
    # Copy factor_engine from local repo (excluding .git)
    rsync -av --exclude='.git' "$SOURCE/" "$TEMP/"
    
    # Commit and push
    (
        cd "$TEMP"
        git add -A
        git commit -m "Sync from quant_projects: $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || echo "No changes to sync"
        git push "https://${HKUST_TOKEN}@github.com/HKUST-QUANT-SOCIETY/factor_engine.git" main
    )
    
    # Cleanup
    rm -rf "$TEMP"
    
    echo "=== HKUST sync complete ==="
fi
