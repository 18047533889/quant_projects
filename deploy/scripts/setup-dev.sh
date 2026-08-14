#!/bin/bash
# Setup script for local development environment

set -e

echo "🚀 Setting up Quant Projects development environment..."

# Check prerequisites
echo ""
echo "1️⃣  Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Please install Docker first."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Please install Docker Compose first."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Please install Python 3.10+."; exit 1; }

echo "   ✅ Docker installed: $(docker --version)"
echo "   ✅ Docker Compose installed: $(docker-compose --version)"
echo "   ✅ Python installed: $(python3 --version)"

# Create necessary directories
echo ""
echo "2️⃣  Creating directories..."
mkdir -p data/quant_workspace
mkdir -p data/cache
mkdir -p logs
echo "   ✅ Directories created"

# Copy environment files
echo ""
echo "3️⃣  Setting up environment files..."
if [ ! -f "deploy/env/.env.local" ]; then
    cp deploy/env/.env.dev deploy/env/.env.local
    echo "   ✅ Created .env.local from template"
    echo "   ⚠️  Please edit deploy/env/.env.local with your local settings"
else
    echo "   ℹ️  .env.local already exists, skipping"
fi

# Install Python development tools
echo ""
echo "4️⃣  Installing Python development tools..."
python3 -m pip install --upgrade pip
pip install ruff black isort mypy pytest pytest-cov

echo "   ✅ Development tools installed"

# Build Docker images
echo ""
echo "5️⃣  Building Docker images..."
cd docker
docker-compose -f docker-compose.dev.yml build --parallel
echo "   ✅ Docker images built"

# Start services
echo ""
echo "6️⃣  Starting services..."
docker-compose -f docker-compose.dev.yml up -d
echo "   ✅ Services started"

# Wait for services to be healthy
echo ""
echo "7️⃣  Waiting for services to be ready..."
sleep 10

# Check DataAccess
if curl -f http://localhost:8765/health >/dev/null 2>&1; then
    echo "   ✅ DataAccess is healthy"
else
    echo "   ⚠️  DataAccess health check failed"
fi

# Check FactorEngine
if curl -f http://localhost:8766/health >/dev/null 2>&1; then
    echo "   ✅ FactorEngine is healthy"
else
    echo "   ⚠️  FactorEngine health check failed"
fi

# Show service URLs
echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Services are running:"
echo "   DataAccess:   http://localhost:8765"
echo "   FactorEngine: http://localhost:8766"
echo "   Prometheus:   http://localhost:9090"
echo "   Grafana:      http://localhost:3000 (admin/admin)"
echo ""
echo "📝 Useful commands:"
echo "   View logs:       cd docker && docker-compose -f docker-compose.dev.yml logs -f"
echo "   Stop services:   cd docker && docker-compose -f docker-compose.dev.yml down"
echo "   Restart:         cd docker && docker-compose -f docker-compose.dev.yml restart"
echo "   Clean up:        cd docker && docker-compose -f docker-compose.dev.yml down -v"
echo ""
echo "📖 See docs/DEPLOYMENT.md for more information"
