# Multi-stage Dockerfile for Llama-4-Scout-17B-16E-Instruct RunPod Serverless
# Optimized for size, security, and cold-start performance

# Stage 1: Base image with system dependencies
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure system to prefer HTTP/1.1 to avoid HTTP/2 framing issues
# Create curl config that forces HTTP/1.1 (only affects curl inside container)
RUN echo "--http1.1" >> /etc/curlrc || true

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Dependencies installation
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Runtime image
FROM dependencies AS runtime

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash runpod && \
    chown -R runpod:runpod /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=runpod:runpod src/ ./src/
COPY --chown=runpod:runpod scripts/ ./scripts/

# Create cache directory in user's home with proper permissions
RUN mkdir -p /home/runpod/.cache/huggingface && \
    mkdir -p /home/runpod/.cache/torch && \
    chown -R runpod:runpod /home/runpod/.cache

# Set environment variables for runtime
ENV PYTHONPATH=/app \
    HF_HOME=/home/runpod/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/runpod/.cache/huggingface \
    HF_DATASETS_CACHE=/home/runpod/.cache/huggingface/datasets \
    TORCH_HOME=/home/runpod/.cache/torch \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    URLLIB3_DISABLE_HTTP2=1

# Enforce SSL certificate verification for Python HTTPS requests (security best practice)
# Expose health check port
EXPOSE 8000

# Health check configuration (Python-based, no external dependencies)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 /app/scripts/healthcheck.py

# Switch to non-root user
USER runpod

# Set entrypoint to start script
CMD ["python", "-u", "src/start.py"]

