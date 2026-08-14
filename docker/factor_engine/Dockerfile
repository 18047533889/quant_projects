# Multi-stage build for FactorEngine service
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
COPY pyproject.toml README.md ./
COPY _build_backend.py ./

# Create minimal package structure for build
RUN mkdir -p api backend cache cleaned_operators docs expr evidence \
    factor_recipes fields ir market mining modeling planner \
    research_tools runtime security semantic service storage util validation

# Copy __init__.py files to satisfy package structure
COPY api/__init__.py api/ || touch api/__init__.py
COPY backend/__init__.py backend/ || touch backend/__init__.py
COPY cache/__init__.py cache/ || touch cache/__init__.py
COPY service/__init__.py service/ || touch service/__init__.py

# Install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels '.[full,performance,service]'

# Final stage
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FACTOR_ENGINE_SERVICE_PORT=8766 \
    PYTHONPATH=/opt/factor_engine \
    FACTOR_ENGINE_BUILD_SHA=${BUILD_SHA:-unknown} \
    FACTOR_ENGINE_BUILD_VERSION=${BUILD_VERSION:-dev}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --uid 10002 factorengine && \
    mkdir -p /opt/factor_engine /data/quant_workspace /tmp/factor_cache && \
    chown -R factorengine:factorengine /opt/factor_engine /data/quant_workspace /tmp/factor_cache

WORKDIR /opt/factor_engine

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install from wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# Copy application code
COPY --chown=factorengine:factorengine . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8766/health', timeout=5).raise_for_status()" || exit 1

USER factorengine
EXPOSE 8766

CMD ["factor-engine-serve", "--host", "0.0.0.0", "--port", "8766"]
