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
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    vim \
    nano \
    less \
    htop \
    ncdu \
    tree \
    zip \
    unzip \
    rsync \
    openssh-client \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install essential build tools
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel \
    build

# Install Python development and testing tools
RUN pip install \
    pytest>=7.4.3 \
    pytest-cov>=4.1.0 \
    pytest-asyncio>=0.21.1 \
    black>=23.12.1 \
    flake8>=6.1.0 \
    mypy>=1.7.1 \
    isort>=5.13.2 \
    pylint>=3.0.3 \
    ipython>=8.18.1 \
    ipdb>=0.13.13

# Set working directory
WORKDIR /workspace

# Pre-install project dependencies for faster startup
# This layer will be cached unless requirements.txt changes
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Create common cache directories
RUN mkdir -p /root/.cache/huggingface && \
    mkdir -p /root/.cache/torch && \
    mkdir -p /root/.cache/pip

# Set environment variables for caching
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
    TORCH_HOME=/root/.cache/torch \
    PYTHONPATH=/workspace

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command: keep container running
CMD ["sleep", "infinity"]

