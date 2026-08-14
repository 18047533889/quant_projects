#!/bin/bash
# Blue-Green Deployment Script for Quant Services

set -euo pipefail

SERVICE_NAME=${1:-}
IMAGE_TAG=${2:-}
NAMESPACE=${NAMESPACE:-quant-production}

if [ -z "$SERVICE_NAME" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <service-name> <image-tag>"
    echo "Example: $0 dataaccess v1.2.3"
    exit 1
fi

echo "🔵 Starting blue-green deployment for $SERVICE_NAME with tag $IMAGE_TAG"

# Determine current active deployment (blue or green)
CURRENT_COLOR=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "📊 Current active: $CURRENT_COLOR, Deploying to: $NEW_COLOR"

# Create new deployment with new color
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}-${NEW_COLOR}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
    color: ${NEW_COLOR}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
      color: ${NEW_COLOR}
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
        color: ${NEW_COLOR}
    spec:
      containers:
      - name: ${SERVICE_NAME}
        image: ghcr.io/\${GITHUB_REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}
        ports:
        - containerPort: 8765
        env:
        - name: DEPLOYMENT_COLOR
          value: ${NEW_COLOR}
        livenessProbe:
          httpGet:
            path: /health
            port: 8765
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8765
          initialDelaySeconds: 10
          periodSeconds: 5
EOF

# Wait for new deployment to be ready
echo "⏳ Waiting for $NEW_COLOR deployment to be ready..."
kubectl rollout status deployment/${SERVICE_NAME}-${NEW_COLOR} -n "$NAMESPACE" --timeout=10m

# Run health checks on new deployment
echo "🔍 Running health checks on $NEW_COLOR deployment..."
NEW_POD=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME",color="$NEW_COLOR" -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n "$NAMESPACE" "$NEW_POD" -- python -c "import requests; requests.get('http://localhost:8765/health').raise_for_status()"

# Switch service to new deployment
echo "🔀 Switching traffic to $NEW_COLOR deployment..."
kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"$SERVICE_NAME\",\"color\":\"$NEW_COLOR\"}}}"

# Monitor for 30 seconds
echo "📡 Monitoring new deployment for 30 seconds..."
sleep 30

# Check for errors
ERROR_COUNT=$(kubectl logs -n "$NAMESPACE" -l app="$SERVICE_NAME",color="$NEW_COLOR" --tail=100 | grep -i "error\|exception\|fatal" | wc -l || echo "0")
if [ "$ERROR_COUNT" -gt 5 ]; then
    echo "❌ High error count detected ($ERROR_COUNT errors). Rolling back..."
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"$SERVICE_NAME\",\"color\":\"$CURRENT_COLOR\"}}}"
    exit 1
fi

# Scale down old deployment
echo "⬇️  Scaling down $CURRENT_COLOR deployment..."
kubectl scale deployment/${SERVICE_NAME}-${CURRENT_COLOR} -n "$NAMESPACE" --replicas=0

echo "✅ Blue-green deployment completed successfully"
echo "   Active deployment: $NEW_COLOR"
echo "   Previous deployment: $CURRENT_COLOR (scaled to 0)"
echo ""
echo "To rollback, run:"
echo "  kubectl patch service $SERVICE_NAME -n $NAMESPACE -p '{\"spec\":{\"selector\":{\"app\":\"$SERVICE_NAME\",\"color\":\"$CURRENT_COLOR\"}}}'"
echo "  kubectl scale deployment/${SERVICE_NAME}-${CURRENT_COLOR} -n $NAMESPACE --replicas=2"
