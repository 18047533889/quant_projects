#!/bin/bash
# Quick reference commands for operator automation tools

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Operator Automation Tools - Quick Reference           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/operator_cli.py" ]; then
    echo "❌ Error: Run this from the factor_engine root directory"
    exit 1
fi

echo "Available Commands:"
echo ""
echo "📊 DIAGNOSTICS"
echo "  ./scripts/operator_cli.py scan                    # Quick scan"
echo "  ./scripts/operator_cli.py stats                   # Show statistics"
echo "  ./scripts/operator_cli.py policy-audit            # Policy gaps"
echo ""
echo "🔧 AUTO-FIX (Preview)"
echo "  ./scripts/operator_cli.py fix-docstrings          # Preview docstring fixes"
echo "  ./scripts/operator_cli.py fix-defaults            # Preview default fixes"
echo ""
echo "✅ AUTO-FIX (Apply)"
echo "  ./scripts/operator_cli.py fix-docstrings --apply  # Apply docstring fixes"
echo "  ./scripts/operator_cli.py fix-defaults --apply    # Apply default fixes"
echo ""
echo "🚀 BATCH OPERATIONS"
echo "  ./scripts/operator_cli.py batch-fix               # Preview batch workflow"
echo "  ./scripts/operator_cli.py batch-fix --git --apply # Apply with git"
echo ""
echo "💬 INTERACTIVE"
echo "  ./scripts/operator_cli.py interactive             # Interactive mode"
echo ""
echo "📖 DOCUMENTATION"
echo "  cat scripts/OPERATOR_TOOLS_README.md              # Full documentation"
echo "  cat scripts/OPERATOR_TOOLS_SUMMARY.md            # Delivery summary"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Offer to run a command
if [ "$1" == "" ]; then
    read -p "Run a command? (scan/stats/fix/batch/interactive/no): " choice
    case $choice in
        scan)
            python3 scripts/operator_cli.py scan
            ;;
        stats)
            python3 scripts/operator_cli.py stats
            ;;
        fix)
            python3 scripts/operator_cli.py fix-docstrings
            ;;
        batch)
            python3 scripts/operator_cli.py batch-fix
            ;;
        interactive)
            python3 scripts/operator_cli.py interactive
            ;;
        no|"")
            echo "👋 Use any command above to get started"
            ;;
        *)
            echo "❌ Unknown option"
            ;;
    esac
fi
