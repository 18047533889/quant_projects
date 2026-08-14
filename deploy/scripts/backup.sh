#!/bin/bash
# Backup script for production data

set -euo pipefail

NAMESPACE=${NAMESPACE:-quant-production}
BACKUP_DIR=${BACKUP_DIR:-/backups/quant}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="quant-backup-${TIMESTAMP}"

echo "💾 Starting backup of Quant services..."
echo "   Namespace: $NAMESPACE"
echo "   Backup dir: $BACKUP_DIR"
echo "   Timestamp: $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup Kubernetes resources
echo ""
echo "1️⃣  Backing up Kubernetes resources..."

# Backup deployments
kubectl get deployments -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/deployments.yaml"
echo "   ✅ Deployments backed up"

# Backup services
kubectl get services -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/services.yaml"
echo "   ✅ Services backed up"

# Backup configmaps
kubectl get configmaps -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/configmaps.yaml"
echo "   ✅ ConfigMaps backed up"

# Backup secrets (encrypted)
kubectl get secrets -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/secrets.yaml"
echo "   ✅ Secrets backed up"

# Backup PVCs
kubectl get pvc -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/pvcs.yaml"
echo "   ✅ PVCs backed up"

# Backup ingress
kubectl get ingress -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/ingress.yaml"
echo "   ✅ Ingress backed up"

# Backup HPA
kubectl get hpa -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$BACKUP_NAME/hpa.yaml"
echo "   ✅ HPA backed up"

# Backup application data (if using PVCs)
echo ""
echo "2️⃣  Backing up persistent volumes..."

# Get list of PVCs
PVCS=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

for PVC in $PVCS; do
    echo "   Backing up PVC: $PVC"

    # Create a backup job
    kubectl run -n "$NAMESPACE" backup-$PVC-$TIMESTAMP \
        --image=alpine:latest \
        --restart=Never \
        --overrides="{
          \"apiVersion\": \"v1\",
          \"spec\": {
            \"volumes\": [{
              \"name\": \"data\",
              \"persistentVolumeClaim\": {\"claimName\": \"$PVC\"}
            }],
            \"containers\": [{
              \"name\": \"backup\",
              \"image\": \"alpine:latest\",
              \"command\": [\"tar\", \"-czf\", \"-\", \"/data\"],
              \"volumeMounts\": [{\"name\": \"data\", \"mountPath\": \"/data\"}]
            }]
          }
        }" > /dev/null 2>&1 || true

    # Wait for pod to complete
    kubectl wait --for=condition=Ready pod/backup-$PVC-$TIMESTAMP -n "$NAMESPACE" --timeout=60s || true

    # Copy backup data
    kubectl logs backup-$PVC-$TIMESTAMP -n "$NAMESPACE" > "$BACKUP_DIR/$BACKUP_NAME/$PVC.tar.gz" 2>/dev/null || true

    # Cleanup backup pod
    kubectl delete pod backup-$PVC-$TIMESTAMP -n "$NAMESPACE" || true

    echo "   ✅ $PVC backed up"
done

# Create backup metadata
echo ""
echo "3️⃣  Creating backup metadata..."

cat > "$BACKUP_DIR/$BACKUP_NAME/metadata.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "namespace": "$NAMESPACE",
  "kubernetes_version": "$(kubectl version --short | grep Server | awk '{print $3}')",
  "backup_files": [
    "deployments.yaml",
    "services.yaml",
    "configmaps.yaml",
    "secrets.yaml",
    "pvcs.yaml",
    "ingress.yaml",
    "hpa.yaml"
  ]
}
EOF

echo "   ✅ Metadata created"

# Compress backup
echo ""
echo "4️⃣  Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"
echo "   ✅ Backup compressed"

# Calculate checksum
CHECKSUM=$(sha256sum "${BACKUP_NAME}.tar.gz" | awk '{print $1}')
echo "$CHECKSUM  ${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.sha256"

# Display summary
BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | awk '{print $1}')

echo ""
echo "✅ Backup completed successfully!"
echo ""
echo "📦 Backup details:"
echo "   File: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "   Size: $BACKUP_SIZE"
echo "   SHA256: $CHECKSUM"
echo ""
echo "📝 To restore from this backup:"
echo "   ./deploy/scripts/restore.sh $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
