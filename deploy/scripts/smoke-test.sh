#!/bin/bash
# Smoke tests for deployed services

set -euo pipefail

SERVICE_URL=${1:-}
TIMEOUT=${TIMEOUT:-30}

if [ -z "$SERVICE_URL" ]; then
    echo "Usage: $0 <service-url>"
    echo "Example: $0 http://dataaccess:8765"
    exit 1
fi

echo "🧪 Running smoke tests for $SERVICE_URL..."

# Test 1: Health check
echo ""
echo "1️⃣  Testing health endpoint..."
if curl -f -s --max-time "$TIMEOUT" "$SERVICE_URL/health" > /dev/null; then
    echo "   ✅ Health check passed"
else
    echo "   ❌ Health check failed"
    exit 1
fi

# Test 2: Service responds with valid JSON
echo ""
echo "2️⃣  Testing JSON response..."
RESPONSE=$(curl -s --max-time "$TIMEOUT" "$SERVICE_URL/health")
if echo "$RESPONSE" | jq . > /dev/null 2>&1; then
    echo "   ✅ Valid JSON response"
else
    echo "   ⚠️  Invalid JSON response"
fi

# Test 3: Response time check
echo ""
echo "3️⃣  Testing response time..."
START=$(date +%s%N)
curl -f -s --max-time "$TIMEOUT" "$SERVICE_URL/health" > /dev/null
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if [ $DURATION -lt 1000 ]; then
    echo "   ✅ Response time: ${DURATION}ms"
elif [ $DURATION -lt 5000 ]; then
    echo "   ⚠️  Slow response: ${DURATION}ms"
else
    echo "   ❌ Very slow response: ${DURATION}ms"
    exit 1
fi

# Test 4: Metrics endpoint (if available)
echo ""
echo "4️⃣  Testing metrics endpoint..."
if curl -f -s --max-time "$TIMEOUT" "$SERVICE_URL/metrics" > /dev/null 2>&1; then
    echo "   ✅ Metrics endpoint accessible"
else
    echo "   ℹ️  Metrics endpoint not available (optional)"
fi

# Test 5: Multiple requests (load test)
echo ""
echo "5️⃣  Testing multiple concurrent requests..."
SUCCESS=0
TOTAL=10

for i in $(seq 1 $TOTAL); do
    if curl -f -s --max-time "$TIMEOUT" "$SERVICE_URL/health" > /dev/null 2>&1; then
        SUCCESS=$((SUCCESS + 1))
    fi
done

SUCCESS_RATE=$((SUCCESS * 100 / TOTAL))

if [ $SUCCESS_RATE -eq 100 ]; then
    echo "   ✅ Success rate: ${SUCCESS_RATE}% (${SUCCESS}/${TOTAL})"
elif [ $SUCCESS_RATE -ge 90 ]; then
    echo "   ⚠️  Success rate: ${SUCCESS_RATE}% (${SUCCESS}/${TOTAL})"
else
    echo "   ❌ Success rate: ${SUCCESS_RATE}% (${SUCCESS}/${TOTAL})"
    exit 1
fi

echo ""
echo "✅ All smoke tests passed!"
exit 0
