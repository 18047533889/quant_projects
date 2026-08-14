# Multi-stage build for FactorOptimizer service
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10004 factoroptimizer && \
    mkdir -p /opt/factor_optimizer /data/workspace && \
    chown -R factoroptimizer:factoroptimizer /opt/factor_optimizer /data/workspace

WORKDIR /opt/factor_optimizer

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

COPY --chown=factoroptimizer:factoroptimizer . .

USER factoroptimizer

CMD ["python", "-m", "factor_optimizer"]
