#!/bin/bash
# Canary Deployment Script for Quant Services

set -euo pipefail

SERVICE_NAME=${1:-}
IMAGE_TAG=${2:-}
NAMESPACE=${NAMESPACE:-quant-production}
CANARY_WEIGHT=${CANARY_WEIGHT:-10}

if [ -z "$SERVICE_NAME" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <service-name> <image-tag>"
    echo "Example: $0 dataaccess v1.2.3"
    exit 1
fi

echo "🐤 Starting canary deployment for $SERVICE_NAME with tag $IMAGE_TAG"
echo "   Canary weight: ${CANARY_WEIGHT}%"

# Create canary deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}-canary
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
    version: canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
      version: canary
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
        version: canary
    spec:
      containers:
      - name: ${SERVICE_NAME}
        image: ghcr.io/\${GITHUB_REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}
        ports:
        - containerPort: 8765
        env:
        - name: DEPLOYMENT_VERSION
          value: canary
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

# Create canary service
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}-canary
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ${SERVICE_NAME}
    version: canary
  ports:
  - port: 8765
    targetPort: 8765
EOF

# Wait for canary deployment
echo "⏳ Waiting for canary deployment to be ready..."
kubectl rollout status deployment/${SERVICE_NAME}-canary -n "$NAMESPACE" --timeout=10m

# Configure traffic splitting (using Istio VirtualService as example)
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
spec:
  hosts:
  - ${SERVICE_NAME}
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: ${SERVICE_NAME}-canary
  - route:
    - destination:
        host: ${SERVICE_NAME}
      weight: $((100 - CANARY_WEIGHT))
    - destination:
        host: ${SERVICE_NAME}-canary
      weight: ${CANARY_WEIGHT}
EOF

echo "📊 Canary receiving ${CANARY_WEIGHT}% of traffic"

# Monitor canary for 5 minutes
echo "📡 Monitoring canary deployment for 5 minutes..."
for i in {1..10}; do
    sleep 30

    # Check error rate
    CANARY_ERRORS=$(kubectl logs -n "$NAMESPACE" -l app="$SERVICE_NAME",version=canary --tail=100 | grep -i "error\|exception" | wc -l || echo "0")
    STABLE_ERRORS=$(kubectl logs -n "$NAMESPACE" -l app="$SERVICE_NAME",version=stable --tail=100 | grep -i "error\|exception" | wc -l || echo "0")

    echo "   Check $i/10: Canary errors=$CANARY_ERRORS, Stable errors=$STABLE_ERRORS"

    if [ "$CANARY_ERRORS" -gt $((STABLE_ERRORS * 2)) ] && [ "$CANARY_ERRORS" -gt 5 ]; then
        echo "❌ Canary error rate too high. Rolling back..."
        kubectl delete deployment ${SERVICE_NAME}-canary -n "$NAMESPACE"
        kubectl delete virtualservice ${SERVICE_NAME} -n "$NAMESPACE"
        exit 1
    fi
done

echo "✅ Canary validation passed. Promoting to stable..."

# Promote canary to stable
kubectl set image deployment/${SERVICE_NAME} ${SERVICE_NAME}=ghcr.io/\${GITHUB_REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG} -n "$NAMESPACE"
kubectl rollout status deployment/${SERVICE_NAME} -n "$NAMESPACE" --timeout=10m

# Clean up canary
kubectl delete deployment ${SERVICE_NAME}-canary -n "$NAMESPACE"
kubectl delete service ${SERVICE_NAME}-canary -n "$NAMESPACE"
kubectl delete virtualservice ${SERVICE_NAME} -n "$NAMESPACE"

echo "✅ Canary deployment promoted successfully"
