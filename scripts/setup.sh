#!/bin/bash
#
# Unified Setup Script for RunPod Llama-4-Scout Serverless Environment
# This script handles project setup, dependency installation, and environment configuration
#
# Version: 1.0

set -euo pipefail

# Script configuration
SCRIPT_VERSION="1.0"
PROJECT_NAME="runpod-meta-Llama-4-Scout-serverless"
DEFAULT_REPO_URL="https://github.com/EcomTree/runpod-meta-Llama-4-Scout-serverless.git"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" >&2
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

# Utility functions
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

retry() {
    local attempt=1
    local exit_code=0
    local max_attempts=${RETRY_ATTEMPTS:-3}
    local delay=${RETRY_DELAY:-2}

    while true; do
        "$@" && return 0
        exit_code=$?

        if (( attempt >= max_attempts )); then
            return "$exit_code"
        fi

        log_warning "Attempt ${attempt}/${max_attempts} failed – retrying in ${delay}s"
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

# Configuration
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    local dir

    if [[ -n "$source" ]]; then
        dir="$(dirname "$source")"
    else
        dir="$(dirname "$0")"
    fi

    if [[ -d "$dir" ]]; then
        (cd "$dir" && pwd)
    else
        pwd
    fi
}

SCRIPT_DIR="$(get_script_dir)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect environment
is_codex_environment() {
    [ -n "${CODEX_CONTAINER:-}" ] || \
    [ -n "${RUNPOD_POD_ID:-}" ] || \
    [ -n "${CODEX_WORKSPACE:-}" ] || \
    [ -d "/workspace" ]
}

# Python dependencies
PYTHON_PACKAGES=("torch" "transformers" "accelerate" "huggingface-hub" "bitsandbytes" \
                 "runpod" "fastapi" "uvicorn" "pydantic" "python-dotenv")

# Check Python version
check_python_version() {
    local required_major=3
    local required_minor=8

    if ! command_exists python3; then
        log_error "Python 3 is not installed"
        return 1
    fi

    local version=$(python3 --version 2>&1 | awk '{print $2}')
    local major=$(echo "$version" | cut -d. -f1)
    local minor=$(echo "$version" | cut -d. -f2)

    log_info "Python Version: $version"

    if ! [[ "$major" =~ ^[0-9]+$ ]] || ! [[ "$minor" =~ ^[0-9]+$ ]]; then
        log_warning "Could not parse version numbers from $version"
        return 0
    fi

    if [ "$major" -lt "$required_major" ] || ([ "$major" -eq "$required_major" ] && [ "$minor" -lt "$required_minor" ]); then
        log_warning "Python $required_major.$required_minor+ recommended, found $version"
        return 0
    fi

    log_success "Python version check passed"
    return 0
}

# Install system packages
install_system_packages() {
    local packages=("$@")
    local missing=()

    for pkg in "${packages[@]}"; do
        if command_exists "$pkg"; then
            log_success "$pkg available"
        else
            missing+=("$pkg")
        fi
    done

    if (( ${#missing[@]} == 0 )); then
        return 0
    fi

    if ! command_exists apt-get; then
        log_warning "apt-get not available – skipping install for (${missing[*]})"
        return 1
    fi

    if command_exists sudo && sudo -n true 2>/dev/null; then
        log_info "Installing packages via sudo apt-get: ${missing[*]}"
        if retry sudo apt-get update -qq; then
            retry sudo apt-get install -y "${missing[@]}"
        else
            log_warning "apt-get update failed – skipping install for (${missing[*]})"
            return 1
        fi
    elif [ "$(id -u)" -eq 0 ]; then
        log_info "Installing packages with root privileges: ${missing[*]}"
        if retry apt-get update -qq; then
            retry apt-get install -y "${missing[@]}"
        else
            log_warning "apt-get update failed – skipping install for (${missing[*]})"
            return 1
        fi
    else
        log_warning "No sudo privileges – cannot install packages (${missing[*]})"
        return 1
    fi

    for pkg in "${missing[@]}"; do
        if command_exists "$pkg"; then
            log_success "$pkg installed"
        else
            log_warning "$pkg installation failed"
        fi
    done
}

# Validate Python packages
validate_python_packages() {
    log_info "Validating Python packages..."

    local packages=("torch" "transformers" "accelerate" "huggingface_hub" "runpod" \
                   "fastapi" "uvicorn" "pydantic" "bitsandbytes")
    local all_ok=true

    for pkg in "${packages[@]}"; do
        if python3 -c "import $pkg" 2>/dev/null; then
            log_success "✓ $pkg"
        else
            log_warning "✗ $pkg not found"
            all_ok=false
        fi
    done

    if $all_ok; then
        log_success "All Python packages validated"
        return 0
    else
        log_warning "Some packages missing - may cause issues"
        return 1
    fi
}

# Setup workspace
setup_workspace() {
    log_info "Setting up workspace..."

    if [ -d "/workspace" ]; then
        cd /workspace
        WORKSPACE_DIR="/workspace"
    else
        if mkdir -p /workspace; then
            cd /workspace
            WORKSPACE_DIR="/workspace"
        else
            log_error "Failed to create /workspace directory. Check permissions."
            exit 1
        fi
    fi

    log_success "Workspace ready: $(pwd)"
}

# Clone or update repository
setup_repository() {
    local repo_url="${1:-$DEFAULT_REPO_URL}"
    local target_dir="${WORKSPACE_DIR}/${PROJECT_NAME}"

    # Check if we're already in the project directory
    if [ -f "src/handler.py" ] && [ -f "requirements.txt" ]; then
        log_info "Already in project directory: $(pwd)"
        PROJECT_ROOT="$(pwd)"
        return 0
    fi

    # Check if project exists in workspace
    if [ -d "$target_dir" ] && [ -f "$target_dir/src/handler.py" ] && [ -f "$target_dir/requirements.txt" ]; then
        log_info "Project directory already exists, skipping clone"
        cd "$target_dir"
        PROJECT_ROOT="$target_dir"
        return 0
    fi

    # Clone repository
    log_info "Cloning repository to $target_dir..."
    if retry git clone "$repo_url" "$target_dir"; then
        cd "$target_dir"
        PROJECT_ROOT="$target_dir"
        log_success "Repository cloned"
    else
        log_error "Failed to clone repository"
        exit 1
    fi
}

# Setup Python environment
setup_python_environment() {
    log_info "Setting up Python environment..."

    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment"
        python3 -m venv .venv
    else
        log_info "Using existing virtual environment"
    fi

    source .venv/bin/activate
    PYTHON_CMD="$(command -v python)"

    log_info "Upgrading pip and core packages"
    retry "$PYTHON_CMD" -m pip install --quiet --upgrade pip setuptools wheel

    log_info "Installing Python dependencies"
    if [ -f "requirements.txt" ]; then
        retry "$PYTHON_CMD" -m pip install --quiet --no-cache-dir -r requirements.txt
    else
        log_warning "requirements.txt not found, installing core packages"
        retry "$PYTHON_CMD" -m pip install --quiet --no-cache-dir "${PYTHON_PACKAGES[@]}"
    fi

    validate_python_packages || log_warning "Package validation failed"
    log_success "Python environment ready"
}

# Setup configuration files
setup_configuration() {
    log_info "Setting up configuration files..."

    # Create .env.example if it doesn't exist
    if [ ! -f ".env.example" ]; then
        cat > .env.example << 'EOF'
# Hugging Face Configuration (REQUIRED)
HF_TOKEN=your_hugging_face_token_here

# Model Configuration
MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct
DEVICE_MAP=auto
TORCH_DTYPE=bfloat16

# Quantization (for lower VRAM)
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false

# Optimization
ENABLE_FLASH_ATTENTION=true

# Inference Configuration
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=50

# Limits
MAX_INPUT_TOKENS=4096
MAX_TOTAL_TOKENS=8192

# Server Configuration
HEALTH_CHECK_HOST=0.0.0.0
HEALTH_CHECK_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
MODEL_WARMUP=true
EOF
        log_success ".env.example created"
    fi

    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_warning "Please edit .env and add your HF_TOKEN!"
    fi

    # Create output directories
    mkdir -p logs
    log_success "Configuration files and directories created"
}

# Setup git configuration
setup_git() {
    log_info "Setting up git configuration..."

    # Configure git user if not set
    if [ -z "$(git config --global user.email 2>/dev/null || true)" ]; then
        git config --global user.email "${GIT_USER_EMAIL:-codex@runpod.io}" 2>/dev/null && \
            log_success "Git email configured" || \
            log_warning "Could not set git email"
    fi

    if [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
        git config --global user.name "${GIT_USER_NAME:-Codex User}" 2>/dev/null && \
            log_success "Git name configured" || \
            log_warning "Could not set git name"
    fi

    git config --global init.defaultBranch main 2>/dev/null && \
        log_success "Git default branch configured" || \
        log_warning "Could not set git default branch"
}

# Validate setup
validate_setup() {
    log_info "Validating setup..."

    # Ensure we're in the project directory
    if [ -n "${PROJECT_ROOT:-}" ] && [ -d "$PROJECT_ROOT" ]; then
        cd "$PROJECT_ROOT"
    fi

    # Check Python syntax
    if [ -f "src/handler.py" ]; then
        if python3 -m py_compile src/handler.py 2>/dev/null; then
            log_success "✓ Python syntax valid"
        else
            log_warning "✗ Python syntax issues detected"
        fi
    else
        log_warning "✗ src/handler.py not found in $(pwd)"
    fi

    # Check if handler can be imported
    if [ -f "src/handler.py" ]; then
        if python3 -c "from src.handler import handler" 2>/dev/null; then
            log_success "✓ Handler importable"
        else
            log_warning "✗ Handler import issues (may need proper environment)"
        fi
    else
        log_warning "✗ Cannot test handler import - src/handler.py not found"
    fi

    # Check GPU availability
    local cuda_check
    local torch_available
    
    # First check if torch is installed
    # Combine torch availability and CUDA check in one Python invocation
    cuda_check=$(python3 - 2>/dev/null <<EOF
try:
    import torch
    print('INSTALLED')
    print(torch.cuda.is_available())
except ImportError:
    print('NOT_INSTALLED')
EOF
    | tail -n 2)
    torch_status=$(echo "$cuda_check" | head -n 1 | tr -d '\r\n')
    cuda_available=$(echo "$cuda_check" | tail -n 1 | tr -d '\r\n')

    if [ "$torch_status" = "INSTALLED" ]; then
        torch_available=true
        if [ "$cuda_available" = "True" ]; then
            log_success "✓ CUDA available"
        else
            log_info "CUDA not available (normal in Codex, required for RunPod deployment)"
        fi
    else
        torch_available=false
        log_warning "✗ PyTorch not installed - cannot check CUDA availability"
    fi

    # Check required files
    local required_files=("src/handler.py" "src/config.py" "requirements.txt" "Dockerfile" "README.md")
    local all_files_ok=true
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "✓ $file"
        else
            log_warning "✗ $file missing in $(pwd)"
            all_files_ok=false
        fi
    done

    if $all_files_ok; then
        log_success "Setup validation completed - all files present"
    else
        log_warning "Setup validation completed with warnings"
    fi
}

# Main setup function
main() {
    log_info "🚀 Starting $PROJECT_NAME setup v$SCRIPT_VERSION"

    # Check environment
    if is_codex_environment; then
        log_success "Codex environment detected"
        export IN_CODEX=true
    else
        log_warning "Not in Codex environment - some features may differ"
        export IN_CODEX=false
    fi

    # Run pre-flight checks
    log_info "🔍 Running pre-flight checks..."
    check_python_version || log_warning "Python version check failed"

    # Setup workspace
    setup_workspace

    # Setup repository
    setup_repository "$@"

    # Setup Python environment
    setup_python_environment

    # Setup configuration
    setup_configuration

    # Setup git
    setup_git

    # Install system packages (optional)
    if [ "$IN_CODEX" = true ]; then
        install_system_packages jq curl git || log_warning "Some system packages not installed"
    fi

    # Validate setup
    validate_setup

    # Show summary
    echo
    log_success "✨ Setup completed successfully!"
    echo
    log_info "📋 Environment Summary:"
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)
    echo "   ├─ Python: ${PYTHON_VERSION:-N/A}"
    PIP_VERSION=$(python3 -m pip --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)
    echo "   ├─ pip: ${PIP_VERSION:-N/A}"
    echo "   ├─ Workspace: $(pwd)"
    echo "   ├─ Virtualenv: $(dirname "$(command -v python 2>/dev/null || echo 'N/A')")"
    
    # Check CUDA
    local cuda_summary
    if python3 -c "import torch" 2>/dev/null; then
        cuda_summary=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | tail -n 1 | tr -d '\r\n')
        if [ "$cuda_summary" = "True" ]; then
            echo "   ├─ CUDA: Available"
            python3 -c "import torch; print(f\"   └─ GPU: {torch.cuda.get_device_name(0)}\")" 2>/dev/null || echo "   └─ GPU: Unknown"
        else
            echo "   └─ CUDA: Not available (GPU required)"
        fi
    else
        echo "   └─ PyTorch: Not installed (cannot check CUDA)"
    fi
    
    echo
    log_info "📝 Next steps:"
    echo "   1. Set HF_TOKEN environment variable in Codex UI"
    echo "   2. Test the handler: python3 -c 'from src.handler import handler'"
    echo "   3. Run tests: python scripts/test_local.py --mode simple"
    echo "   4. For Docker build: docker build -t llama4-scout-runpod:latest ."
    echo

    if [ "$IN_CODEX" = true ]; then
        log_info "💡 Codex-specific tips:"
        echo "   • Enable 'Container Caching' for faster restarts"
        echo "   • Set HF_TOKEN in environment variables"
        echo "   • Test locally before deploying to RunPod"
        echo "   • Monitor GPU usage: nvidia-smi"
        echo "   • Reference: https://docs.runpod.io/"
        echo
    fi

    log_success "🎉 Environment is ready!"
}

# Run main function with all arguments
main "$@"
