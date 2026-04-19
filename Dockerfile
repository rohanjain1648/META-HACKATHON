# ForgeRL Environment — Docker Container
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY forge_env/requirements.txt ./forge_env/requirements.txt
COPY forgeai/requirements.txt ./forgeai/requirements.txt

RUN pip install --no-cache-dir -r forge_env/requirements.txt
RUN pip install --no-cache-dir -r forgeai/requirements.txt

# Copy source code
COPY forge_env/ ./forge_env/
COPY forgeai/ ./forgeai/
COPY README.md .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for OpenEnv API
EXPOSE 7860

# Run the OpenEnv server
CMD ["uvicorn", "forge_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
