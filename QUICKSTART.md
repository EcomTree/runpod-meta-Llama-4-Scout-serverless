# Quick Start Guide

Get your Llama-4-Scout model running in 5 minutes!

## 🚀 Super Quick Start (Local Testing)

```bash
# 1. Set your Hugging Face token
export HF_TOKEN="your_token_here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a test
python scripts/test_local.py --mode simple
```

## 🐳 Docker Quick Start

```bash
# 1. Build the image
docker build -t llama4-scout:test .

# 2. Run it (requires GPU)
docker run --gpus all \
  -e HF_TOKEN="your_token" \
  -p 8000:8000 \
  llama4-scout:test

# 3. Test health endpoint (in another terminal)
curl http://localhost:8000/health
```

## ☁️ RunPod Deployment (5 Steps)

### 1. Build & Push

```bash
export DOCKER_USERNAME="your-dockerhub-username"

docker build -t $DOCKER_USERNAME/llama4-scout-runpod:latest .
docker push $DOCKER_USERNAME/llama4-scout-runpod:latest
```

### 2. Create RunPod Endpoint

- Go to: https://www.runpod.io/console/serverless
- Click "New Endpoint"
- Docker Image: `your-username/llama4-scout-runpod:latest`
- GPU: A5000 (24GB)

### 3. Add Environment Variables

```
HF_TOKEN=your_hugging_face_token
```

### 4. Configure Health Check

- Path: `/health`
- Port: `8000`
- Start Period: `300` seconds

### 5. Deploy & Test

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/runsync" \
  -H "Authorization: Bearer YOUR-API-KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is AI?",
      "max_new_tokens": 100
    }
  }'
```

## 🎯 Example Request

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/YOUR-ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR-KEY"},
    json={
        "input": {
            "prompt": "Explain quantum computing",
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }
)

print(response.json()["output"]["generated_text"])
```

## 📚 Need More Details?

- **Full Setup**: See [README.md](README.md)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## ⚡ Performance Tips

- Use `bfloat16` for best performance on A5000+
- Enable Flash Attention: `ENABLE_FLASH_ATTENTION=true`
- For GPUs < 24GB: `LOAD_IN_8BIT=true`

## 🆘 Having Issues?

1. Check logs in RunPod dashboard
2. Verify HF_TOKEN is correct
3. Ensure model access on Hugging Face
4. See [README.md](README.md) troubleshooting section

---

**That's it!** You're ready to generate text with Llama-4-Scout! 🎉

