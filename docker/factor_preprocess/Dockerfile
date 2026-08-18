# Multi-stage build for FactorPreprocess service
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY factor_preprocess/ factor_preprocess/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10003 factorpreprocess && \
    mkdir -p /opt/factor_preprocess /data/workspace && \
    chown -R factorpreprocess:factorpreprocess /opt/factor_preprocess /data/workspace

WORKDIR /opt/factor_preprocess

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

USER factorpreprocess

# This package has no service or CLI; keep the compose dependency container alive
# without inventing runtime semantics until an explicit worker contract exists.
CMD ["tail", "-f", "/dev/null"]
