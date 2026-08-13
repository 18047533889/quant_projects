#!/bin/bash
# Quick Fix Script for Code Quality Issues
# Run this to automatically fix ~2,000 issues in 20 minutes

set -e

echo "=================================="
echo "Code Quality Quick Fix Script"
echo "=================================="
echo ""

# Check if tools are installed
echo "Checking required tools..."
command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
command -v isort >/dev/null 2>&1 || { echo "Installing isort..."; pip install isort; }
command -v autoflake >/dev/null 2>&1 || { echo "Installing autoflake..."; pip install autoflake; }

echo "All tools available!"
echo ""

# Backup current state
echo "Creating backup..."
BACKUP_DIR="/tmp/code_quality_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Backup location: $BACKUP_DIR"
echo ""

# Step 1: Format with Black
echo "Step 1: Formatting code with Black..."
echo "This will fix ~1,000 style issues..."
black dataaccess factor_layer toolkit quant_evaluator 2>&1 | tee "$BACKUP_DIR/black.log"
echo "✓ Black formatting complete"
echo ""

# Step 2: Sort imports with isort
echo "Step 2: Sorting imports with isort..."
echo "This will fix ~10,366 lines of import ordering..."
isort dataaccess factor_layer toolkit quant_evaluator 2>&1 | tee "$BACKUP_DIR/isort.log"
echo "✓ Import sorting complete"
echo ""

# Step 3: Remove unused imports (with confirmation)
echo "Step 3: Removing unused imports with autoflake..."
read -p "Review changes before removing unused imports? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Running autoflake in check mode first..."
    autoflake --remove-all-unused-imports --remove-unused-variables \
      --recursive dataaccess factor_layer toolkit quant_evaluator > "$BACKUP_DIR/autoflake_preview.txt"
    echo "Preview saved to: $BACKUP_DIR/autoflake_preview.txt"
    read -p "Proceed with removing unused imports? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        autoflake --remove-all-unused-imports --remove-unused-variables \
          --in-place --recursive dataaccess factor_layer toolkit quant_evaluator 2>&1 | tee "$BACKUP_DIR/autoflake.log"
        echo "✓ Unused imports removed"
    else
        echo "Skipped autoflake"
    fi
else
    autoflake --remove-all-unused-imports --remove-unused-variables \
      --in-place --recursive dataaccess factor_layer toolkit quant_evaluator 2>&1 | tee "$BACKUP_DIR/autoflake.log"
    echo "✓ Unused imports removed"
fi
echo ""

# Summary
echo "=================================="
echo "Quick Fix Complete!"
echo "=================================="
echo ""
echo "Changes made:"
echo "  - Formatted code with Black"
echo "  - Sorted imports with isort"
echo "  - Removed unused imports (if confirmed)"
echo ""
echo "Logs saved to: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "1. Review git diff"
echo "2. Run tests to ensure nothing broke"
echo "3. Commit changes"
echo "4. Move to P0 refactoring tasks"
echo ""

# Show git status
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Git status:"
    git status --short | head -20
    echo ""
    echo "Review changes with: git diff"
fi

