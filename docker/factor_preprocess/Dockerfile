# Multi-stage build for FactorPreprocess service
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

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

COPY --chown=factorpreprocess:factorpreprocess . .

USER factorpreprocess

CMD ["python", "-m", "factor_preprocess"]
