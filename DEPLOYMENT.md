# Deployment Guide - Llama-4-Scout-17B-16E-Instruct on RunPod

Complete step-by-step guide for deploying to RunPod Serverless.

## Prerequisites Checklist

- [ ] RunPod account created and verified
- [ ] Hugging Face account with access to Llama-4-Scout model
- [ ] HF_TOKEN generated with read access
- [ ] Docker installed locally (for building)
- [ ] Docker Hub (or other registry) account

## Step 1: Prepare Your Environment

### 1.1 Clone and Setup

```bash
cd /path/to/your/workspace
cd runpod-meta-Llama-4-Scout-serverless

# Copy environment template
cp .env.example .env

# Edit .env and add your HF_TOKEN
nano .env
```

### 1.2 Verify Access to Model

```bash
# Test that your token has access
export HF_TOKEN="your_token_here"

python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info('meta-llama/Llama-4-Scout-17B-16E-Instruct', token='$HF_TOKEN')
    print('✓ Access granted to model')
except Exception as e:
    print(f'✗ Access denied: {e}')
"
```

If access is denied, request it at: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

## Step 2: Local Testing (Optional but Recommended)

### 2.1 Test Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run simple test
python scripts/test_local.py --mode simple
```

### 2.2 Test With Docker Locally

```bash
# Build image
docker build -t llama4-scout-test .

# Run with GPU
docker run --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  llama4-scout-test

# In another terminal, test health endpoint
curl http://localhost:8000/health
```

## Step 3: Build and Push Docker Image

### 3.1 Build Production Image

```bash
# Set your Docker Hub username
export DOCKER_USERNAME="your-dockerhub-username"

# Build the image
docker build -t $DOCKER_USERNAME/llama4-scout-runpod:latest .

# Test the image locally one more time
docker run --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  $DOCKER_USERNAME/llama4-scout-runpod:latest
```

### 3.2 Push to Registry

```bash
# Login to Docker Hub
docker login

# Push the image
docker push $DOCKER_USERNAME/llama4-scout-runpod:latest

# Verify it's uploaded
docker pull $DOCKER_USERNAME/llama4-scout-runpod:latest
```

**Alternative: Using Makefile**

```bash
# Set variables
export DOCKER_USERNAME="your-dockerhub-username"

# Build and push in one command
make deploy DOCKER_USERNAME=$DOCKER_USERNAME
```

## Step 4: Create RunPod Serverless Endpoint

### 4.1 Access RunPod Dashboard

1. Go to https://www.runpod.io/console/serverless
2. Click "**New Endpoint**"

### 4.2 Configure Endpoint

#### Basic Settings

- **Name**: `llama4-scout-17b-inference`
- **Select GPU Type**: 
  - Recommended: **A5000** (24GB VRAM) or higher
  - Minimum: **A4000** (16GB VRAM) with 8-bit quantization
- **Docker Image**: `your-dockerhub-username/llama4-scout-runpod:latest`

#### Container Configuration

- **Container Disk**: `20 GB` (for system and dependencies)
- **Volume Disk**: `50 GB` (for model caching, highly recommended)
- **Volume Mount Path**: `/root/.cache/huggingface`

#### Environment Variables

Click "**Add Environment Variable**" for each:

```
HF_TOKEN=your_hugging_face_token_here
MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct
TORCH_DTYPE=bfloat16
ENABLE_FLASH_ATTENTION=true
LOG_LEVEL=INFO
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_TEMPERATURE=0.7
MODEL_WARMUP=true
```

**For Memory-Constrained GPUs (< 24GB):**

```
LOAD_IN_8BIT=true
TORCH_DTYPE=float16
```

#### Worker Configuration

- **Max Workers**: `1` (recommended for large models)
- **Min Workers**: `0` (scale to zero when idle)
- **Idle Timeout**: `60` seconds
- **Execution Timeout**: `120` seconds
- **Max Concurrent Requests**: `1`

#### Health Check Configuration

- **Health Check Enabled**: ✓ (checked)
- **Health Check Path**: `/health`
- **Health Check Port**: `8000`
- **Health Check Timeout**: `10` seconds
- **Start Period**: `300` seconds (5 minutes for model loading)

#### Advanced Settings (Optional)

- **Network Storage**: Enable if you want to persist model across deployments
- **Allow Public Access**: Enable if you want public access without API key

### 4.3 Deploy

1. Review all settings
2. Click "**Deploy**"
3. Wait for status to change from "Initializing" to "Running"
   - This can take 5-10 minutes for the first deployment
   - Model needs to download (~34GB)
4. Monitor logs in the RunPod dashboard

## Step 5: Get Your Endpoint Details

After deployment:

1. Click on your endpoint in the dashboard
2. Note down:
   - **Endpoint ID**: `xxxxxxxxxx-xxxxxxxxxx`
   - **API Key**: Found in "Settings" tab

## Step 6: Test Your Deployment

### 6.1 Using cURL

```bash
# Set your credentials
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

# Test the endpoint
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Explain artificial intelligence in simple terms.",
      "max_new_tokens": 200,
      "temperature": 0.7
    }
  }'
```

### 6.2 Using Python

Create `test_runpod.py`:

```python
import requests
import os

endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
api_key = os.getenv("RUNPOD_API_KEY")

url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

response = requests.post(
    url,
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "input": {
            "prompt": "Write a haiku about artificial intelligence.",
            "max_new_tokens": 100,
            "temperature": 0.8,
        }
    },
    timeout=120
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

Run it:

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
python test_runpod.py
```

## Step 7: Monitor Your Deployment

### 7.1 Check Logs

In RunPod dashboard:
1. Go to your endpoint
2. Click "**Logs**" tab
3. Monitor for errors or warnings

### 7.2 Check Health

```bash
# Note: Health endpoint is internal to RunPod, not publicly accessible
# But you can see health status in the dashboard
```

### 7.3 Monitor Costs

1. Go to "**Billing**" in RunPod dashboard
2. Check GPU usage and costs
3. Adjust idle timeout if needed to minimize costs

## Step 8: Optimization Tips

### 8.1 Reduce Cold Start Time

**Option 1: Keep Workers Warm**
- Set `Min Workers: 1` (costs more but eliminates cold starts)

**Option 2: Use Network Storage**
- Enable network storage volume
- Model persists across deployments

**Option 3: Pre-bake Model into Image**
- Uncomment model download stage in Dockerfile
- Image will be larger (~40GB) but cold start faster

### 8.2 Optimize for Throughput

```env
# Increase max workers if you have traffic
MAX_WORKERS=3

# Adjust timeouts
IDLE_TIMEOUT=30
EXECUTION_TIMEOUT=60
```

### 8.3 Optimize for Cost

```env
# Scale to zero when idle
MIN_WORKERS=0
IDLE_TIMEOUT=60

# Use quantization to allow smaller GPUs
LOAD_IN_8BIT=true
```

## Troubleshooting Common Issues

### Issue: Container Keeps Restarting

**Check:**
1. View logs for error messages
2. Verify HF_TOKEN is correct
3. Ensure GPU has enough memory
4. Check model access on Hugging Face

### Issue: Slow Response Times

**Solutions:**
1. Enable Flash Attention: `ENABLE_FLASH_ATTENTION=true`
2. Use bfloat16: `TORCH_DTYPE=bfloat16`
3. Reduce max tokens: `DEFAULT_MAX_NEW_TOKENS=256`

### Issue: Out of Memory Errors

**Solutions:**
1. Enable 8-bit quantization: `LOAD_IN_8BIT=true`
2. Use larger GPU type (A5000 or A6000)
3. Reduce batch size or max tokens

### Issue: Health Check Failing

**Check:**
1. Increase start period to 600 seconds
2. Verify port 8000 is exposed
3. Check health endpoint path is `/health`

## Cost Estimation

Approximate costs on RunPod (as of 2024):

| GPU | VRAM | $/hour | Idle $/hour | Monthly (24/7) |
|-----|------|--------|-------------|----------------|
| A4000 | 16GB | $0.34 | $0.03 | ~$245 |
| A5000 | 24GB | $0.49 | $0.04 | ~$353 |
| A6000 | 48GB | $0.79 | $0.05 | ~$569 |

**With Auto-Scaling (Min Workers = 0):**
- Only pay when processing requests
- Typical API usage: $10-50/month

## Security Checklist

- [ ] HF_TOKEN stored in RunPod secrets (not in code)
- [ ] API key rotated regularly
- [ ] Public access disabled (unless needed)
- [ ] Logs monitored for suspicious activity
- [ ] HTTPS used for all API calls

## Next Steps

1. **Set up monitoring**: Use RunPod webhooks for alerts
2. **Create staging environment**: Test changes before production
3. **Implement rate limiting**: Add API gateway if needed
4. **Set up CI/CD**: Automate builds and deployments
5. **Monitor costs**: Set up billing alerts

## Support Resources

- **RunPod Docs**: https://docs.runpod.io
- **RunPod Discord**: https://discord.gg/runpod
- **Support**: support@runpod.io
- **Status Page**: https://status.runpod.io

---

**Congratulations!** 🎉 Your Llama-4-Scout model is now deployed and ready for production use!

