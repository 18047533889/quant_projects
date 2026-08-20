FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    FACTOR_ENGINE_SERVICE_PORT=8766 \
    PYTHONPATH=/opt/factor_engine

RUN useradd --create-home --uid 10002 factorengine \
    && mkdir -p /opt/factor_engine /data/quant_workspace \
    && chown -R factorengine:factorengine /opt/factor_engine /data/quant_workspace

WORKDIR /opt/factor_engine

# Install system dependencies for numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --no-cache-dir '.[full,performance,service]'

USER factorengine
EXPOSE 8766

CMD ["factor-engine-serve", "--host", "0.0.0.0", "--port", "8766"]
