#!/bin/bash
# Rollback Script for Quant Services

set -euo pipefail

SERVICE_NAME=${1:-}
NAMESPACE=${NAMESPACE:-quant-production}
ROLLBACK_REVISION=${2:-}

if [ -z "$SERVICE_NAME" ]; then
    echo "Usage: $0 <service-name> [revision]"
    echo "Example: $0 dataaccess"
    echo "         $0 dataaccess 5"
    exit 1
fi

echo "🔄 Starting rollback for $SERVICE_NAME in namespace $NAMESPACE"

# Show current deployment status
echo "📊 Current deployment status:"
kubectl get deployment "$SERVICE_NAME" -n "$NAMESPACE" -o wide

# Show rollout history
echo ""
echo "📜 Rollout history:"
kubectl rollout history deployment/"$SERVICE_NAME" -n "$NAMESPACE"

# Determine rollback target
if [ -n "$ROLLBACK_REVISION" ]; then
    echo ""
    echo "⏪ Rolling back to revision $ROLLBACK_REVISION..."
    kubectl rollout undo deployment/"$SERVICE_NAME" -n "$NAMESPACE" --to-revision="$ROLLBACK_REVISION"
else
    echo ""
    echo "⏪ Rolling back to previous revision..."
    kubectl rollout undo deployment/"$SERVICE_NAME" -n "$NAMESPACE"
fi

# Wait for rollback to complete
echo ""
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/"$SERVICE_NAME" -n "$NAMESPACE" --timeout=10m

# Verify pods are healthy
echo ""
echo "🔍 Verifying pod health..."
READY_PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -o "True" | wc -l)
TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" --no-headers | wc -l)

echo "   Ready pods: $READY_PODS/$TOTAL_PODS"

if [ "$READY_PODS" -lt "$TOTAL_PODS" ]; then
    echo "❌ Not all pods are ready after rollback"
    kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME"
    exit 1
fi

# Check recent logs for errors
echo ""
echo "📋 Checking recent logs for errors..."
ERROR_COUNT=$(kubectl logs -n "$NAMESPACE" -l app="$SERVICE_NAME" --tail=100 --since=2m | grep -i "error\|exception\|fatal" | wc -l || echo "0")
echo "   Error count in last 2 minutes: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "⚠️  Warning: High error count detected after rollback"
fi

echo ""
echo "✅ Rollback completed successfully"
echo ""
echo "Current deployment:"
kubectl get deployment "$SERVICE_NAME" -n "$NAMESPACE" -o wide
echo ""
echo "Running pods:"
kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME"
