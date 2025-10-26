# Llama-4-Scout-17B-16E-Instruct RunPod Serverless Deployment

Production-ready serverless deployment solution for the **meta-llama/Llama-4-Scout-17B-16E-Instruct** model on RunPod infrastructure.

## 🚀 Features

- **Optimized Cold Start**: Model loads once at container startup
- **GPU Acceleration**: Full CUDA support with optional Flash Attention 2
- **Memory Efficient**: Support for FP16/BF16 and quantization (4-bit/8-bit)
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Health Checks**: Built-in FastAPI health endpoints for container monitoring
- **Input Validation**: Robust input validation with Pydantic
- **Metrics Collection**: Detailed performance metrics for every request

## 📋 Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Hugging Face Token**: Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Requires access to `meta-llama/Llama-4-Scout-17B-16E-Instruct`
3. **Docker**: For local testing and building
4. **CUDA-capable GPU**: A5000 or higher recommended (24GB+ VRAM)

## 🏗️ Project Structure

```text
.
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Central configuration management
│   ├── utils.py              # Utilities, logging, exceptions
│   ├── model_loader.py       # Singleton model loader
│   ├── handler.py            # RunPod handler (main entry point)
│   ├── health_server.py      # FastAPI health check server
│   └── start.py              # Startup script
├── tests/
│   └── test_handler.py       # Unit tests
├── scripts/
│   ├── setup.sh              # Codex environment setup script
│   ├── common-codex.sh       # Shared Codex helpers
│   ├── healthcheck.py        # Health check script
│   └── test_local.py         # Local testing script
├── docker/
│   └── .dockerignore         # Docker ignore patterns
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Local development setup
├── requirements.txt          # Python dependencies
├── CODEX_SETUP.md            # Codex environment setup guide
└── README.md                 # This file
```

## 🔧 Local Development & Testing

### Codex Setup (Recommended)

For quick setup in RunPod Codex or similar cloud environments:

1. **Go to Codex Setup Script** and paste:
```bash
curl -fsSL https://raw.githubusercontent.com/EcomTree/runpod-meta-Llama-4-Scout-serverless/main/scripts/setup.sh | bash
```

2. **Set Environment Variables** in Codex UI:
   - `HF_TOKEN`: Your Hugging Face token (required)
   - `MODEL_ID`: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
   - `TORCH_DTYPE`: `bfloat16`
   - `ENABLE_FLASH_ATTENTION`: `true`

3. **See [CODEX_SETUP.md](CODEX_SETUP.md) for detailed instructions**

### Local Setup

#### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/EcomTree/runpod-meta-Llama-4-Scout-serverless.git
cd runpod-meta-Llama-4-Scout-serverless

# Set your Hugging Face token
# ⚠️ Security tip: To avoid saving your token in shell history:
#   Option 1: Load from secure file (recommended)
#     export HF_TOKEN=$(cat ~/.hf_token)
#   Option 2: Use 'read -s' for secure input
#     read -s HF_TOKEN && export HF_TOKEN
#   Option 3: Prefix with space (bash with HISTCONTROL=ignorespace)
export HF_TOKEN="your_hf_token_here"

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Local Testing (Without Docker)

Test the handler locally without containerization:

```bash
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

### 3. Run Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python tests/test_handler.py
```

### 4. Local Docker Testing

Test the complete Docker setup locally:

```bash
# Build the Docker image
docker build -t llama4-scout-runpod:latest .

# Run with Docker Compose
docker-compose up

# Or run directly
docker run --gpus all \
  -e HF_TOKEN="your_token_here" \
  -p 8000:8000 \
  llama4-scout-runpod:latest

# Check health endpoint
curl http://localhost:8000/health
```

## 🚢 RunPod Deployment

### Step 1: Build and Push Docker Image

```bash
# Login to Docker Hub (or your registry)
docker login

# Build the image
docker build -t your-dockerhub-username/llama4-scout-runpod:latest .

# Push to registry
docker push your-dockerhub-username/llama4-scout-runpod:latest
```

### Step 2: Create RunPod Serverless Endpoint

1. **Go to RunPod Dashboard**: [runpod.io/console/serverless](https://www.runpod.io/console/serverless)

2. **Create New Endpoint**:
   - Click "New Endpoint"
   - Select "Custom" template

3. **Configure Endpoint**:
   - **Name**: `llama4-scout-17b`
   - **Docker Image**: `your-dockerhub-username/llama4-scout-runpod:latest`
   - **GPU**: Select A5000 or higher (24GB+ VRAM)
   - **Container Disk**: 20GB minimum
   - **Volume Disk**: 50GB recommended (for model caching)

4. **Environment Variables**:
   ```env
   HF_TOKEN=your_hugging_face_token
   MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct
   TORCH_DTYPE=bfloat16
   ENABLE_FLASH_ATTENTION=true
   LOG_LEVEL=INFO
   DEFAULT_MAX_NEW_TOKENS=512
   ```

5. **Advanced Settings**:
   - **Max Workers**: 1 (for large models)
   - **Idle Timeout**: 60 seconds
   - **Execution Timeout**: 120 seconds
   - **Health Check Path**: `/health`
   - **Health Check Port**: `8000`

6. **Deploy**: Click "Deploy" and wait for initialization

### Step 3: Test Your Endpoint

Using cURL:

```bash
# Set environment variables securely
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-runpod-api-key"

curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is artificial intelligence?",
      "max_new_tokens": 200,
      "temperature": 0.7
    }
  }'
```

Using Python:

```python
import requests
import os

# Load credentials from environment variables
endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
api_key = os.getenv("RUNPOD_API_KEY")

if not endpoint_id or not api_key:
    raise ValueError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables must be set")

response = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "input": {
            "prompt": "Explain quantum computing in simple terms.",
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
)

result = response.json()
print(result)
```

## 📝 API Reference

### Input Schema

```json
{
  "input": {
    "prompt": "string (required)",
    "max_new_tokens": "integer (optional, default: 512, range: 1-8192)",
    "temperature": "float (optional, default: 0.7, range: 0.0-2.0)",
    "top_p": "float (optional, default: 0.9, range: 0.0-1.0)",
    "top_k": "integer (optional, default: 50, range: 0+)",
    "repetition_penalty": "float (optional, default: 1.1, range: 1.0-2.0)",
    "do_sample": "boolean (optional, default: true)"
  }
}
```

### Success Response

```json
{
  "output": {
    "generated_text": "string",
    "tokens_generated": 256,
    "input_tokens": 12,
    "total_tokens": 268,
    "generation_time_ms": 2543,
    "total_time_ms": 2567,
    "tokenization_time_ms": 15,
    "decoding_time_ms": 9,
    "tokens_per_second": 100.67
  },
  "request_id": "req_1234567890"
}
```

### Error Response

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Input validation failed: prompt cannot be empty",
    "timestamp": "2024-01-15T10:30:00.000Z"
  },
  "request_id": "req_1234567891"
}
```

## ⚙️ Configuration

All configuration is done via environment variables:

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Model identifier |
| `HF_TOKEN` | *required* | Hugging Face authentication token |
| `DEVICE_MAP` | `auto` | Device mapping strategy |
| `TORCH_DTYPE` | `bfloat16` | Model dtype (auto, float16, bfloat16) |
| `LOAD_IN_8BIT` | `false` | Enable 8-bit quantization |
| `LOAD_IN_4BIT` | `false` | Enable 4-bit quantization |
| `ENABLE_FLASH_ATTENTION` | `true` | Use Flash Attention 2 if available |

### Inference Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Default max tokens to generate |
| `DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `DEFAULT_TOP_P` | `0.9` | Default nucleus sampling parameter |
| `DEFAULT_TOP_K` | `50` | Default top-k sampling parameter |
| `MAX_INPUT_TOKENS` | `4096` | Maximum input token length |
| `MAX_TOTAL_TOKENS` | `8192` | Maximum total tokens (input + output) |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HEALTH_CHECK_HOST` | `0.0.0.0` | Health check server host |
| `HEALTH_CHECK_PORT` | `8000` | Health check server port |
| `MODEL_WARMUP` | `true` | Perform warmup inference on startup |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `json` | Log format (json, text) |

## 🔍 Monitoring & Health Checks

### Health Endpoints

- **`GET /health`**: Overall health status (returns 200 if ready, 503 if not)
- **`GET /ready`**: Readiness probe (returns 200 only when model is loaded)
- **`GET /liveness`**: Liveness probe (returns 200 if container is alive)
- **`GET /metrics`**: Performance metrics

### Example Health Check Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_ready": true,
  "model_load_time_seconds": 45.23,
  "gpu": {
    "device": "NVIDIA RTX A5000",
    "memory_allocated_gb": 18.45,
    "memory_free_gb": 5.55,
    "memory_total_gb": 24.0
  }
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Fails to Start

**Symptom**: Container exits immediately or health check never passes

**Solutions**:
- Check logs: `docker logs <container_id>`
- Verify HF_TOKEN is set correctly
- Ensure sufficient GPU memory (24GB+ recommended)
- Check model access permissions on Hugging Face

#### 2. Out of Memory (OOM) Errors

**Symptom**: "CUDA out of memory" errors

**Solutions**:
```bash
# Enable 8-bit quantization
export LOAD_IN_8BIT=true

# Or use 4-bit quantization (even lower memory)
export LOAD_IN_4BIT=true

# Reduce default max tokens
export DEFAULT_MAX_NEW_TOKENS=256
```

#### 3. Slow Cold Start

**Symptom**: Container takes too long to become ready

**Solutions**:
- Use volume mounting for model cache
- Consider pre-baking model into Docker image (increases image size)
- Use faster storage tier on RunPod

#### 4. Authentication Errors

**Symptom**: "401 Unauthorized" or "Repository not found"

**Solutions**:
- Verify HF_TOKEN has correct permissions
- Request access to Llama-4-Scout model on Hugging Face
- Check token hasn't expired

#### 5. Generation Quality Issues

**Symptom**: Poor or repetitive outputs

**Solutions**:
```python
# Adjust generation parameters
{
  "temperature": 0.8,  # Increase for more creativity
  "top_p": 0.95,       # Adjust nucleus sampling
  "repetition_penalty": 1.2  # Increase to reduce repetition
}
```

### Debugging Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
```

## 📊 Performance Optimization

### Recommended Settings by GPU

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| A5000 | 24GB | `bfloat16`, no quantization |
| A4000 | 16GB | `LOAD_IN_8BIT=true` |
| RTX 4090 | 24GB | `bfloat16`, Flash Attention 2 |

### Tips for Faster Inference

1. **Enable Flash Attention 2** (already enabled by default)
2. **Use bfloat16** for better performance than float16
3. **Reduce max_new_tokens** for shorter responses
4. **Batch requests** if using multiple workers
5. **Use volume caching** to persist model weights

## 🔒 Security Best Practices

1. **Never commit tokens**: Use environment variables
2. **Rotate tokens regularly**: Update HF_TOKEN periodically
3. **Use secrets management**: Store tokens in RunPod secrets
4. **Monitor access**: Check logs for suspicious activity
5. **Update dependencies**: Keep packages up to date

## 📄 License

This deployment code is provided under the MIT License. Note that the Llama-4-Scout model itself is subject to Meta's license terms.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## 📞 Support

- **Issues**: Open an issue on GitHub
- **RunPod Support**: [support.runpod.io](https://support.runpod.io)
- **Hugging Face**: [discuss.huggingface.co](https://discuss.huggingface.co)

## 🙏 Acknowledgments

- Meta AI for the Llama-4-Scout model
- RunPod for serverless infrastructure
- Hugging Face for model hosting and transformers library

---

**Happy deploying!** 🚀
