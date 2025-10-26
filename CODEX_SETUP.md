# Codex Environment Setup Guide

## 🎯 Overview

This document describes how to set up the RunPod Llama-4-Scout Serverless repo in the Codex environment.

## 🚀 Quick Start

### In Codex Web UI:

1. **Insert Setup Script:**
   - Go to Codex → "Setup Script"
   - Select "Manual"
   - Paste the following command:

```bash
# Codex Setup for RunPod Llama-4-Scout Serverless
curl -fsSL https://raw.githubusercontent.com/EcomTree/runpod-meta-Llama-4-Scout-serverless/main/scripts/setup.sh | bash
```

OR (if you want to test a branch):

```bash
# Run Setup Script
git clone https://github.com/EcomTree/runpod-meta-Llama-4-Scout-serverless.git /workspace/runpod-llama4-scout-serverless
cd /workspace/runpod-llama4-scout-serverless
bash scripts/setup.sh
```

2. **Set Environment Variables (Required):**
   - Click on "Environment Variables" → "Add"
   - Add the following variables:

   | Variable | Value | Description |
   |----------|------|-------------|
   | `HF_TOKEN` | `your-token` | **Required** - Hugging Face token for model access |
   | `MODEL_ID` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Model identifier |
   | `TORCH_DTYPE` | `bfloat16` | Model dtype (auto, float16, bfloat16) |
   | `ENABLE_FLASH_ATTENTION` | `true` | Use Flash Attention 2 |
   | `LOG_LEVEL` | `INFO` | Logging level |
   | `DEFAULT_MAX_NEW_TOKENS` | `512` | Default max tokens to generate |

3. **Start Container:**
   - Enable "Container Caching" (optional but recommended)
   - Start the environment

## 📦 What Gets Installed?

The setup script automatically installs:

### Python Packages:
- ✅ `torch` - PyTorch with CUDA support
- ✅ `transformers` - Hugging Face Transformers library
- ✅ `accelerate` - Hugging Face Accelerate
- ✅ `bitsandbytes` - Model quantization
- ✅ `runpod` - RunPod SDK
- ✅ `fastapi` - Web server for health checks
- ✅ `uvicorn` - ASGI server
- ✅ `pydantic` - Data validation
- ✅ `python-dotenv` - Environment variable management

### System Tools:
- ✅ `jq` - JSON Parser (for debugging)
- ✅ `curl` - HTTP Client
- ✅ `git` - Version control

### Already Pre-installed (according to Codex):
- ✅ Python 3.12
- ✅ Node.js 20
- ✅ CUDA Toolkit (GPU support)
- ✅ Various development tools

## 🔧 Configuration

### Model Configuration

This setup is optimized for the **Llama-4-Scout-17B-16E-Instruct** model:

```bash
# Required
HF_TOKEN=your_hugging_face_token

# Model settings
MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct
TORCH_DTYPE=bfloat16
DEVICE_MAP=auto

# Optimization
ENABLE_FLASH_ATTENTION=true
LOAD_IN_8BIT=false  # Enable if low on VRAM
LOAD_IN_4BIT=false  # Enable for 16GB VRAM GPUs
```

### Inference Configuration

```bash
# Generation parameters
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=50

# Limits
MAX_INPUT_TOKENS=4096
MAX_TOTAL_TOKENS=8192
```

### Server Configuration

```bash
# Health check server
HEALTH_CHECK_HOST=0.0.0.0
HEALTH_CHECK_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 🧪 Testing in Codex

After setup, you can test the following in Codex:

```bash
# In Codex Terminal:
cd /workspace/runpod-llama4-scout-serverless

# Test Python Syntax
python3 -m py_compile src/handler.py

# Check Dependencies
python3 -c "import torch, transformers, runpod; print('✅ All dependencies available')"

# Check GPU
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Test Handler Import
python3 -c "from src.handler import handler; print('✅ Handler importable')"

# Check GPU Memory
python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

## 🚀 Local Testing (Without Docker)

Test the handler locally:

```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Simple test
python scripts/test_local.py --mode simple

# Custom prompt test
python scripts/test_local.py --mode custom \
  --prompt "Explain quantum computing" \
  --max-tokens 200 \
  --temperature 0.8

# Edge case testing
python scripts/test_local.py --mode edge

# Health check test
python scripts/test_local.py --mode health

# Performance benchmark
python scripts/test_local.py --mode benchmark --num-requests 10
```

## 📝 API Testing

### Test Handler Function

```python
# test_handler.py
from src.handler import handler

event = {
    "input": {
        "prompt": "What is artificial intelligence?",
        "max_new_tokens": 200,
        "temperature": 0.7
    }
}

result = handler(event)
print(result)
```

### Health Check

```bash
# Start health server
python src/health_server.py

# In another terminal
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

## 🐛 Troubleshooting

### "Connection Error" in Codex Terminal

This is normal on first start. The setup script creates the necessary structure automatically.

### "HF_TOKEN not configured"

You must set the `HF_TOKEN` environment variable in Codex UI.

### Python Module not found

```bash
# Run setup again:
cd /workspace/runpod-llama4-scout-serverless
bash scripts/setup.sh
```

### CUDA/GPU errors

```bash
# Check GPU
nvidia-smi

# Verify CUDA version
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"

# Check PyTorch GPU support
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### Model loading fails

```bash
# Check Hugging Face token
python3 -c "from huggingface_hub import whoami; print(whoami())"

# Check model access
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.model_info('meta-llama/Llama-4-Scout-17B-16E-Instruct'))"
```

### Out of Memory errors

For GPUs with less than 24GB VRAM:

```bash
# Enable 8-bit quantization
export LOAD_IN_8BIT=true

# Or use 4-bit quantization (even lower memory)
export LOAD_IN_4BIT=true

# Reduce default max tokens
export DEFAULT_MAX_NEW_TOKENS=256
```

## 🎯 Next Steps

After successful setup:

1. **Local Testing:**
   ```bash
   # Test the handler
   python3 -c "from src.handler import handler; print('✅ Handler importable')"
   ```

2. **Run Local Tests:**
   ```bash
   # Run unit tests
   python -m pytest tests/

   # Test locally
   python scripts/test_local.py --mode simple
   ```

3. **Docker Build (for Deployment):**
   ```bash
   docker build -t your-username/llama4-scout-runpod:latest .
   ```

4. **RunPod Deployment:**
   - Push the image to Docker Hub
   - Create Serverless Endpoint in RunPod
   - Configure environment variables
   - Test the endpoint

## 💡 Tips

- ✅ **Enable Container Caching** in Codex for faster starts
- ✅ **Use Hugging Face Token** with proper permissions
- ✅ **Monitor GPU usage** with `nvidia-smi` or `watch -n 1 nvidia-smi`
- ✅ **Test locally** before deploying to RunPod
- ✅ **Check model access** on Hugging Face before setup
- ✅ **Use quantization** for lower VRAM GPUs (A4000, RTX 4090)

## 🔍 GPU Requirements

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| A5000 | 24GB | `bfloat16`, no quantization |
| A4000 | 16GB | `LOAD_IN_8BIT=true` |
| RTX 4090 | 24GB | `bfloat16`, Flash Attention 2 |
| RTX 3090 | 24GB | `bfloat16`, Flash Attention 2 |

## 📊 Performance

Expected performance on A5000 (24GB):
- Model Load Time: ~45-60 seconds
- Cold Start: ~60-90 seconds
- Tokens per Second: ~8-12 tokens/s
- Memory Usage: ~18-20GB

## 🆘 Support

For questions or problems:
- Check the logs: `cat /workspace/logs/*.log`
- GitHub Issues: https://github.com/EcomTree/runpod-meta-Llama-4-Scout-serverless/issues
- RunPod Docs: https://docs.runpod.io/
- Hugging Face: https://huggingface.co/meta-llama

---

**Created for Codex Environment Setup** 🚀
