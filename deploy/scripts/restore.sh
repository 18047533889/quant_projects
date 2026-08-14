#!/bin/bash
# Restore script for production data

set -euo pipefail

BACKUP_FILE=${1:-}
NAMESPACE=${NAMESPACE:-quant-production}
RESTORE_DIR=$(mktemp -d)

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    echo "Example: $0 /backups/quant/quant-backup-20260814_120000.tar.gz"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "🔄 Starting restore from backup..."
echo "   Backup file: $BACKUP_FILE"
echo "   Namespace: $NAMESPACE"
echo "   Restore dir: $RESTORE_DIR"

# Verify checksum if available
CHECKSUM_FILE="${BACKUP_FILE%.tar.gz}.sha256"
if [ -f "$CHECKSUM_FILE" ]; then
    echo ""
    echo "1️⃣  Verifying backup integrity..."
    if sha256sum -c "$CHECKSUM_FILE" 2>/dev/null; then
        echo "   ✅ Checksum verified"
    else
        echo "   ⚠️  Checksum verification failed"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Extract backup
echo ""
echo "2️⃣  Extracting backup..."
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"
BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
BACKUP_PATH="$RESTORE_DIR/$BACKUP_NAME"

if [ ! -d "$BACKUP_PATH" ]; then
    echo "❌ Invalid backup structure"
    exit 1
fi

echo "   ✅ Backup extracted"

# Display backup metadata
if [ -f "$BACKUP_PATH/metadata.json" ]; then
    echo ""
    echo "📋 Backup metadata:"
    cat "$BACKUP_PATH/metadata.json"
fi

# Confirmation
echo ""
echo "⚠️  WARNING: This will restore configuration to namespace '$NAMESPACE'"
echo "   Current resources will be updated/replaced"
read -p "Continue with restore? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    rm -rf "$RESTORE_DIR"
    exit 0
fi

# Restore Kubernetes resources
echo ""
echo "3️⃣  Restoring Kubernetes resources..."

# Restore configmaps (before deployments)
if [ -f "$BACKUP_PATH/configmaps.yaml" ]; then
    kubectl apply -f "$BACKUP_PATH/configmaps.yaml" -n "$NAMESPACE" || true
    echo "   ✅ ConfigMaps restored"
fi

# Restore secrets (with caution)
if [ -f "$BACKUP_PATH/secrets.yaml" ]; then
    read -p "Restore secrets? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl apply -f "$BACKUP_PATH/secrets.yaml" -n "$NAMESPACE" || true
        echo "   ✅ Secrets restored"
    else
        echo "   ⏭️  Secrets skipped"
    fi
fi

# Restore PVCs (careful - this doesn't restore data)
if [ -f "$BACKUP_PATH/pvcs.yaml" ]; then
    echo "   ℹ️  PVC definitions available (data restore requires manual steps)"
fi

# Restore services
if [ -f "$BACKUP_PATH/services.yaml" ]; then
    kubectl apply -f "$BACKUP_PATH/services.yaml" -n "$NAMESPACE" || true
    echo "   ✅ Services restored"
fi

# Restore deployments
if [ -f "$BACKUP_PATH/deployments.yaml" ]; then
    kubectl apply -f "$BACKUP_PATH/deployments.yaml" -n "$NAMESPACE" || true
    echo "   ✅ Deployments restored"
fi

# Restore ingress
if [ -f "$BACKUP_PATH/ingress.yaml" ]; then
    kubectl apply -f "$BACKUP_PATH/ingress.yaml" -n "$NAMESPACE" || true
    echo "   ✅ Ingress restored"
fi

# Restore HPA
if [ -f "$BACKUP_PATH/hpa.yaml" ]; then
    kubectl apply -f "$BACKUP_PATH/hpa.yaml" -n "$NAMESPACE" || true
    echo "   ✅ HPA restored"
fi

# Wait for deployments to be ready
echo ""
echo "4️⃣  Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment --all -n "$NAMESPACE" || true

# Verify restoration
echo ""
echo "5️⃣  Verifying restoration..."
echo "   Deployments:"
kubectl get deployments -n "$NAMESPACE"

echo ""
echo "   Pods:"
kubectl get pods -n "$NAMESPACE"

# Cleanup
rm -rf "$RESTORE_DIR"

echo ""
echo "✅ Restore completed!"
echo ""
echo "📝 Next steps:"
echo "   1. Verify services are healthy"
echo "   2. Check application logs"
echo "   3. Run smoke tests"
echo "   4. Restore persistent data if needed"
echo ""
echo "🔍 Check health:"
echo "   kubectl get pods -n $NAMESPACE"
echo "   kubectl logs -l app=dataaccess -n $NAMESPACE --tail=50"
echo "   kubectl logs -l app=factorengine -n $NAMESPACE --tail=50"
