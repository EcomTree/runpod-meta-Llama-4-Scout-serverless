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
# NOTE: Only critical packages (build-essential, git, ca-certificates) are version-pinned with wildcards
#       to balance reproducibility and build stability. Utilities use latest available to avoid failures
#       when Ubuntu removes older patch versions from repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9.* \
    git=1:2.34.* \
    ca-certificates=20230311.* \
    curl \
    wget \
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

# Upgrade pip and install essential build tools + development/testing tools
# Consolidated into single RUN to reduce image layers
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build && \
    pip install --no-cache-dir \
    pytest>=7.4.3,<8.0 \
    pytest-cov>=4.1.0,<5.0 \
    pytest-asyncio>=0.21.1,<1.0 \
    black>=23.12.1,<24.0 \
    flake8>=6.1.0,<7.0 \
    mypy>=1.7.1,<2.0 \
    isort>=5.13.2,<6.0 \
    pylint>=3.0.3,<4.0 \
    ipython>=8.18.1,<9.0 \
    ipdb>=0.13.13,<1.0

# Set working directory
WORKDIR /workspace

# Pre-install project dependencies for faster startup
# This layer will be cached unless requirements.txt changes
# NOTE: Build context must be project root, not .cursor/ directory
#       Example: docker build -f .cursor/Dockerfile -t my-image .
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt && \
    mkdir -p /workspace/.cache/huggingface && \
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

