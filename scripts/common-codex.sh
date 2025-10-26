#!/bin/bash

# Shared helpers for Codex setup scripts.

if [[ -z "${CODEX_COMMON_HELPERS_LOADED:-}" ]]; then
    CODEX_COMMON_HELPERS_LOADED=1

    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'

    echo_info() {
        echo -e "${BLUE}ℹ️  $1${NC}"
    }

    echo_success() {
        echo -e "${GREEN}✅ $1${NC}"
    }

    echo_warning() {
        echo -e "${YELLOW}⚠️  $1${NC}" >&2
    }

    echo_error() {
        echo -e "${RED}❌ $1${NC}" >&2
    }

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

            echo_warning "Attempt ${attempt}/${max_attempts} failed – retrying in ${delay}s"
            sleep "$delay"
            attempt=$((attempt + 1))
        done
    }

    ensure_system_packages() {
        local packages=("$@")
        local missing=()

        for pkg in "${packages[@]}"; do
            if command_exists "$pkg"; then
                echo_success "$pkg available"
            else
                missing+=("$pkg")
            fi
        done

        if (( ${#missing[@]} == 0 )); then
            return 0
        fi

        if ! command_exists apt-get; then
            echo_warning "apt-get not available – skipping install for (${missing[*]})"
            return 1
        fi

        if command_exists sudo && sudo -n true 2>/dev/null; then
            echo_info "Installing packages via sudo apt-get: ${missing[*]}"
            if retry sudo apt-get update -qq; then
                retry sudo apt-get install -y "${missing[@]}"
            else
                echo_warning "apt-get update failed – skipping install for (${missing[*]})"
                return 1
            fi
        elif [ "$(id -u)" -eq 0 ]; then
            echo_info "Installing packages with root privileges: ${missing[*]}"
            if retry apt-get update -qq; then
                retry apt-get install -y "${missing[@]}"
            else
                echo_warning "apt-get update failed – skipping install for (${missing[*]})"
                return 1
            fi
        else
            echo_warning "No sudo privileges – cannot install packages (${missing[*]})"
            return 1
        fi

        for pkg in "${missing[@]}"; do
            if command_exists "$pkg"; then
                echo_success "$pkg installed"
            else
                echo_warning "$pkg installation failed"
            fi
        done
    }

    resolve_path() {
        local path="$1"
        if [[ -z "$path" ]]; then
            return 1
        fi
        if [[ "$path" == /* ]]; then
            printf '%s\n' "$path"
        else
            printf '%s/%s\n' "$(pwd)" "$path"
        fi
    }

    is_codex_environment() {
        [ -n "${CODEX_CONTAINER:-}" ] || \
        [ -n "${RUNPOD_POD_ID:-}" ] || \
        [ -n "${CODEX_WORKSPACE:-}" ] || \
        [ -d "/workspace" ]
    }

    check_gpu_availability() {
        if command_exists nvidia-smi; then
            echo_info "GPU Information:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        else
            echo_warning "nvidia-smi not found - GPU may not be available"
        fi
    }

    check_cuda_support() {
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
            echo_success "CUDA is available"
            
            python3 -c "import torch; print(f\"  PyTorch: {torch.__version__}\")" 2>/dev/null
            python3 -c "import torch; print(f\"  CUDA Version: {torch.version.cuda}\")" 2>/dev/null || true
            python3 -c "import torch; print(f\"  GPU Count: {torch.cuda.device_count()}\")" 2>/dev/null || true
        else
            echo_warning "CUDA is not available - GPU support will be limited"
        fi
    }

    validate_python_dependencies() {
        echo_info "Checking Python dependencies..."
        
        local packages=("torch" "transformers" "accelerate" "runpod" "fastapi" "pydantic")
        local all_ok=true

        for pkg in "${packages[@]}"; do
            if python3 -c "import $pkg" 2>/dev/null; then
                echo_success "  ✓ $pkg"
            else
                echo_error "  ✗ $pkg not found"
                all_ok=false
            fi
        done

        if $all_ok; then
            echo_success "All core dependencies available"
            return 0
        else
            echo_warning "Some dependencies missing"
            return 1
        fi
    }

    check_hf_token() {
        if [ -n "${HF_TOKEN:-}" ]; then
            echo_success "HF_TOKEN is configured"
            
            # Validate token (basic check)
            if python3 -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
                echo_success "HF_TOKEN is valid"
            else
                echo_warning "HF_TOKEN may be invalid or expired"
            fi
        else
            echo_warning "HF_TOKEN is not set - required for model access"
            echo_info "Set it via: export HF_TOKEN=your_token"
        fi
    }
fi
