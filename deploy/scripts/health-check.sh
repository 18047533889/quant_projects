#!/bin/bash
# Health check script for Quant Services

set -euo pipefail

SERVICE_NAME=${1:-}
SERVICE_URL=${2:-}

if [ -z "$SERVICE_NAME" ] || [ -z "$SERVICE_URL" ]; then
    echo "Usage: $0 <service-name> <service-url>"
    echo "Example: $0 dataaccess http://dataaccess:8765"
    exit 1
fi

echo "🔍 Running health checks for $SERVICE_NAME"
echo "   URL: $SERVICE_URL"

# Basic health endpoint check
echo ""
echo "1️⃣  Checking health endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health" || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Health check passed (HTTP $HTTP_CODE)"
else
    echo "   ❌ Health check failed (HTTP $HTTP_CODE)"
    exit 1
fi

# Response time check
echo ""
echo "2️⃣  Checking response time..."
RESPONSE_TIME=$(curl -s -o /dev/null -w "%{time_total}" "$SERVICE_URL/health")
RESPONSE_MS=$(echo "$RESPONSE_TIME * 1000" | bc)

echo "   Response time: ${RESPONSE_MS}ms"

if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
    echo "   ⚠️  Warning: Response time over 5 seconds"
fi

# Metrics endpoint check (if available)
echo ""
echo "3️⃣  Checking metrics endpoint..."
METRICS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/metrics" || echo "000")

if [ "$METRICS_CODE" = "200" ]; then
    echo "   ✅ Metrics endpoint available"
elif [ "$METRICS_CODE" = "404" ]; then
    echo "   ℹ️  Metrics endpoint not available (404)"
else
    echo "   ⚠️  Metrics endpoint returned HTTP $METRICS_CODE"
fi

# Memory check via metrics (if Prometheus format)
if [ "$METRICS_CODE" = "200" ]; then
    echo ""
    echo "4️⃣  Checking resource metrics..."

    METRICS_OUTPUT=$(curl -s "$SERVICE_URL/metrics")

    # Extract memory usage if available
    MEMORY_BYTES=$(echo "$METRICS_OUTPUT" | grep "process_resident_memory_bytes" | grep -v "#" | awk '{print $2}' | head -1 || echo "0")
    if [ "$MEMORY_BYTES" != "0" ]; then
        MEMORY_MB=$(echo "scale=2; $MEMORY_BYTES / 1024 / 1024" | bc)
        echo "   Memory usage: ${MEMORY_MB}MB"
    fi

    # Extract request count if available
    REQUEST_COUNT=$(echo "$METRICS_OUTPUT" | grep "http_requests_total" | grep -v "#" | awk '{print $2}' | head -1 || echo "0")
    if [ "$REQUEST_COUNT" != "0" ]; then
        echo "   Total requests: $REQUEST_COUNT"
    fi
fi

echo ""
echo "✅ All health checks completed"
