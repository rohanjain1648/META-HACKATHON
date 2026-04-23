# ForgeAI-RL — OpenEnv SDLC Environment
# HuggingFace Spaces Dockerfile (Docker SDK)
#
# This builds a lightweight container that serves the SDLCEnvironment
# as an OpenEnv-compatible FastAPI REST API.
#
# The environment does NOT require GPU — it runs sandboxed Python
# subprocesses for test execution.  Training happens on the client side.

FROM python:3.11-slim

WORKDIR /app

# System dependencies for subprocess-based test execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install only the server dependencies (not the heavy GPU training stack)
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.0.0 \
    pytest>=8.0.0 \
    "pytest-timeout>=2.3.0" \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    rich>=13.0.0 \
    click>=8.0.0 \
    openenv>=0.1.0 \
    fastmcp>=0.1.0

# Copy project source
COPY forgeai/ /app/forgeai/
COPY app.py /app/app.py

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 hfuser && chown -R hfuser:hfuser /app
USER hfuser

# HF Spaces exposes port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "app.py"]
