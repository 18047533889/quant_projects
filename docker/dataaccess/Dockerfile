# Multi-stage build for DataAccess service
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
COPY pyproject.toml README.md ./
COPY core/__init__.py core/
COPY registry/__init__.py registry/
COPY read/__init__.py read/
COPY write/__init__.py write/
COPY cos/__init__.py cos/
COPY service/__init__.py service/
COPY clickhouse/__init__.py clickhouse/
COPY quality/__init__.py quality/
COPY contract/__init__.py contract/
COPY security/__init__.py security/
COPY runtime/__init__.py runtime/
COPY snapshot/__init__.py snapshot/
COPY r30/__init__.py r30/

# Install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels '.[service,polars,sql]'

# Final stage
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_ACCESS_API_HOST=0.0.0.0 \
    DATA_ACCESS_API_PORT=8765 \
    DUCKDB_TEMP_DIRECTORY=/tmp/duckdb \
    DATA_ACCESS_BUILD_SHA=${BUILD_SHA:-unknown} \
    DATA_ACCESS_BUILD_VERSION=${BUILD_VERSION:-dev}

# Create non-root user
RUN useradd --create-home --uid 10001 dataaccess && \
    mkdir -p /tmp/duckdb /var/lib/dataaccess /opt/dataaccess && \
    chown -R dataaccess:dataaccess /tmp/duckdb /var/lib/dataaccess /opt/dataaccess

WORKDIR /opt/dataaccess

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install from wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# Copy application code
COPY --chown=dataaccess:dataaccess . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8765/health', timeout=5).raise_for_status()" || exit 1

USER dataaccess
EXPOSE 8765

CMD ["data-access-server", "--host", "0.0.0.0", "--port", "8765"]
