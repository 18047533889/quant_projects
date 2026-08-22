FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DATA_ACCESS_API_HOST=0.0.0.0 \
    DATA_ACCESS_API_PORT=8765 \
    DUCKDB_TEMP_DIRECTORY=/tmp/duckdb

RUN useradd --create-home --uid 10001 dataaccess \
    && mkdir -p /tmp/duckdb /var/lib/dataaccess \
    && chown -R dataaccess:dataaccess /tmp/duckdb /var/lib/dataaccess

WORKDIR /opt/dataaccess
COPY constraints/py310.txt /tmp/constraints.txt
COPY . .
RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt '.[service]'

USER dataaccess
EXPOSE 8765
CMD ["data-access-server", "--host", "0.0.0.0", "--port", "8765"]
