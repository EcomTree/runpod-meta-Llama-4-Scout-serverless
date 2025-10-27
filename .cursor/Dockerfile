# Cursor Background Agent Dockerfile
# Development environment for Llama-4-Scout RunPod Serverless project

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variables for optimal Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Install essential system dependencies and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9* \
    git=1:2.34.* \
    curl=7.81.* \
    wget=1.21.* \
    ca-certificates \
    vim=2:8.2.* \
    nano=6.2.* \
    less=590.* \
    htop=3.0.* \
    ncdu=1.16.* \
    tree=2.0.* \
    zip=3.0.* \
    unzip=6.0.* \
    rsync=3.2.* \
    openssh-client=1:8.9.* \
    software-properties-common=0.99.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install essential build tools + development/testing tools
# Consolidated into single RUN to reduce image layers
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build && \
    pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    pytest-asyncio==0.21.1 \
    black==23.12.1 \
    flake8==6.1.0 \
    mypy==1.7.1 \
    isort==5.13.2 \
    pylint==3.0.3 \
    ipython==8.18.1 \
    ipdb==0.13.13

# Set working directory
WORKDIR /workspace

# Pre-install project dependencies for faster startup
# This layer will be cached unless requirements.txt changes
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Create common cache directories in a location accessible to all users
RUN mkdir -p /workspace/.cache/huggingface && \
    mkdir -p /workspace/.cache/torch && \
    mkdir -p /workspace/.cache/pip

# Set environment variables for caching
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TORCH_HOME=/workspace/.cache/torch \
    PYTHONPATH=/workspace

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command: keep container running
CMD ["sleep", "infinity"]

