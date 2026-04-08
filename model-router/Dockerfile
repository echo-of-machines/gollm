FROM python:3.12-slim

# Install Docker CLI + Compose plugin (for compose-up fallback in router.py)
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.32.1 \
    httpx==0.28.1 \
    docker==7.1.0 \
    pyyaml==6.0.2 \
    "ruamel.yaml==0.18.6" \
    "huggingface_hub==0.27.0"

WORKDIR /app
COPY router.py config.yaml ./
# static/ is mounted at runtime via compose volume (./model-router/static:/app/static:ro)
# so it doesn't need to be baked in — but create the dir so the mount point always exists
RUN mkdir -p /app/static

# Health check — hits the router's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:30000/health')"

EXPOSE 30000

CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "30000", "--log-level", "info", "--no-access-log"]
