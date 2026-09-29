# aurum: dashboard, MCP server and trading runner in one image.
#   docker compose up -d          (see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AURUM_HOME=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY aurum ./aurum
COPY tradingview ./tradingview
RUN pip install --no-cache-dir ".[mcp,llm,data]" \
 && useradd --create-home --uid 10001 aurum \
 && mkdir -p /data && chown aurum /data
USER aurum
VOLUME ["/data"]
EXPOSE 8765
CMD ["python", "-m", "aurum.web", "--host", "0.0.0.0", "--port", "8765"]
