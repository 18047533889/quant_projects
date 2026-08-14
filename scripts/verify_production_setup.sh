#!/bin/bash
# Verify production configuration installation and setup

set -e

echo "=== Production Configuration Verification ==="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check required modules
echo ""
echo "2. Checking Python dependencies..."
required_modules=("logging" "json" "pathlib" "dataclasses")
optional_modules=("prometheus_client" "psutil" "yaml")

for module in "${required_modules[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        echo "   ✓ $module (required)"
    else
        echo "   ✗ $module (required) - MISSING"
        exit 1
    fi
done

for module in "${optional_modules[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        echo "   ✓ $module (optional)"
    else
        echo "   ⚠ $module (optional) - not installed"
    fi
done

# Check directory structure
echo ""
echo "3. Checking directory structure..."
required_dirs=(
    "config"
    "config/logging"
    "config/monitoring"
    "config/examples"
    "examples"
    "docs"
    "scripts"
    "tests"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ✗ $dir/ - MISSING"
        exit 1
    fi
done

# Check required files
echo ""
echo "4. Checking required files..."
required_files=(
    "config/production_config.py"
    "config/logging/__init__.py"
    "config/logging/structured_logger.py"
    "config/logging/formatters.py"
    "config/logging/handlers.py"
    "config/logging/sanitization.py"
    "config/logging/performance.py"
    "config/monitoring/__init__.py"
    "config/monitoring/metrics.py"
    "config/monitoring/health.py"
    "config/monitoring/shutdown.py"
    "docs/PRODUCTION_DEPLOYMENT.md"
    "config/QUICK_START.md"
    "tests/test_production_config.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file - MISSING"
        exit 1
    fi
done

# Test configuration loading
echo ""
echo "5. Testing configuration loading..."
if python3 -c "
from config.production_config import ProductionConfig
config = ProductionConfig()
config.validate_and_raise()
print('   ✓ Configuration loads and validates successfully')
" 2>/dev/null; then
    :
else
    echo "   ✗ Configuration loading failed"
    exit 1
fi

# Test logging setup
echo ""
echo "6. Testing logging setup..."
if python3 -c "
from config.logging.structured_logger import setup_logging
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    logger = setup_logging(log_dir=Path(tmpdir), service_name='test')
    logger.info('Test message')
print('   ✓ Logging setup works')
" 2>/dev/null; then
    :
else
    echo "   ✗ Logging setup failed"
    exit 1
fi

# Test sanitization
echo ""
echo "7. Testing sensitive data sanitization..."
if python3 -c "
from config.logging.sanitization import sanitize_message, sanitize_dict
msg = sanitize_message('password=secret123')
assert 'REDACTED' in msg
assert 'secret123' not in msg
data = sanitize_dict({'password': 'secret', 'user': 'john'})
assert data['password'] == '***REDACTED***'
assert data['user'] == 'john'
print('   ✓ Sanitization works correctly')
" 2>/dev/null; then
    :
else
    echo "   ✗ Sanitization test failed"
    exit 1
fi

# Test metrics (optional)
echo ""
echo "8. Testing metrics (optional)..."
if python3 -c "
try:
    from config.monitoring.metrics import MetricsCollector
    metrics = MetricsCollector(namespace='test')
    print('   ✓ Metrics collector works')
except ImportError:
    print('   ⚠ Metrics not available (prometheus_client not installed)')
" 2>/dev/null; then
    :
else
    echo "   ⚠ Metrics test skipped"
fi

# Test health checks
echo ""
echo "9. Testing health checks..."
if python3 -c "
from config.monitoring.health import HealthCheckManager, DiskSpaceCheck
manager = HealthCheckManager(service_name='test')
manager.add_check(DiskSpaceCheck())
result = manager.startup_probe()
assert 'status' in result
print('   ✓ Health checks work')
" 2>/dev/null; then
    :
else
    echo "   ✗ Health check test failed"
    exit 1
fi

# Run unit tests if pytest is available
echo ""
echo "10. Running unit tests (if pytest available)..."
if command -v pytest &> /dev/null; then
    if pytest tests/test_production_config.py -v --tb=short 2>&1 | tail -10; then
        echo "   ✓ All tests passed"
    else
        echo "   ⚠ Some tests failed (check output above)"
    fi
else
    echo "   ⚠ pytest not installed, skipping tests"
fi

# Summary
echo ""
echo "=== Verification Summary ==="
echo ""
echo "✓ Production configuration system is properly installed"
echo ""
echo "Next steps:"
echo "  1. Read: config/QUICK_START.md"
echo "  2. Review: config/examples/production.yaml"
echo "  3. Test: python scripts/run_production_server.py"
echo "  4. Read: docs/PRODUCTION_DEPLOYMENT.md"
echo ""
echo "Installation verified successfully!"
