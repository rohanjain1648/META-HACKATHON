# ForgeRL — HuggingFace Spaces Dockerfile (Docker SDK fallback)
#
# Serves the ForgeRL environment as a Gradio + FastAPI app.
# Preferred deployment: use the Gradio SDK (requirements.txt + app.py).
# This Dockerfile is provided for Docker-SDK Spaces and local Docker runs.

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest pytest-timeout

# Copy project source
COPY forgeai/   /app/forgeai/
COPY forge_env/ /app/forge_env/
COPY app.py     /app/app.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# HF Spaces runs as non-root
RUN useradd -m -u 1000 hfuser && chown -R hfuser:hfuser /app
USER hfuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["python", "app.py"]
