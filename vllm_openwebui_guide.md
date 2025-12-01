# Complete Guide: Setting Up vLLM with Qwen Model and Open WebUI on Ubuntu 22.04

This guide provides detailed steps for setting up vLLM with a Qwen model in Docker Compose, integrated with Open WebUI for a beautiful chat interface.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Install Docker](#step-1-install-docker)
- [Step 2: Install NVIDIA Container Toolkit](#step-2-install-nvidia-container-toolkit)
- [Step 3: Create Project Directory](#step-3-create-project-directory-structure)
- [Step 4: Create Docker Compose Configuration](#step-4-create-docker-compose-configuration)
- [Step 5: Choose Your Qwen Model](#step-5-choose-your-qwen-model)
- [Step 6: Configure Hugging Face Token (Optional)](#step-6-optional---configure-hugging-face-token)
- [Step 7: Start Services](#step-7-start-vllm-and-open-webui-services)
- [Step 8: Access Open WebUI](#step-8-access-open-webui)
- [Step 9: Test the Setup](#step-9-test-the-setup)
- [Management Commands](#step-10-useful-management-commands)
- [Advanced Configuration](#advanced-configuration-options)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

First, ensure your system meets these requirements:

### Check NVIDIA GPU and Drivers

```bash
# Check if you have an NVIDIA GPU
lspci | grep -i nvidia

# Check NVIDIA driver version (should be 525.x or newer)
nvidia-smi
```

### Install NVIDIA Drivers (if needed)

```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

---

## Step 1: Install Docker

### Remove Old Docker Versions

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### Install Docker Engine

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### Verify Docker Installation

```bash
docker --version
docker compose version
```

---

## Step 2: Install NVIDIA Container Toolkit

This allows Docker to access your GPU:

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 3: Create Project Directory Structure

```bash
# Create project directory
mkdir -p ~/vllm-qwen
cd ~/vllm-qwen

# Create subdirectories
mkdir -p models logs open-webui
```

---

## Step 4: Create Docker Compose Configuration

Create a `docker-compose.yml` file:

```bash
nano docker-compose.yml
```

Add the following complete configuration with both vLLM and Open WebUI:

```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - HF_HOME=/root/.cache/huggingface
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface
      - ./logs:/logs
    shm_size: '16gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0
      --port 8000
      --max-model-len 4096
      --gpu-memory-utilization 0.9
      --served-model-name qwen-7b
    restart: unless-stopped
    networks:
      - ai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - ./open-webui:/app/backend/data
    environment:
      - OPENAI_API_BASE_URLS=http://vllm:8000/v1
      - OPENAI_API_KEYS=sk-dummy-key
      - WEBUI_AUTH=False
      - WEBUI_NAME=Qwen Chat
      - DEFAULT_MODELS=qwen-7b
    restart: unless-stopped
    networks:
      - ai-network
    depends_on:
      vllm:
        condition: service_healthy

networks:
  ai-network:
    driver: bridge
```

### Configuration Explanation

**vLLM Service:**
- `image`: Official vLLM Docker image with OpenAI-compatible API
- `runtime: nvidia`: Enables GPU access
- `ports`: Maps container port 8000 to host port 8000
- `volumes`: Persists model cache and logs
- `shm_size`: Shared memory size (important for large models)
- `command`: Runs the vLLM OpenAI API server with Qwen model

**Open WebUI Service:**
- `image`: Official Open WebUI image
- `ports`: Web interface accessible on port 3000
- `environment`: Configured to connect to vLLM service
- `depends_on`: Waits for vLLM to be healthy before starting

---

## Step 5: Choose Your Qwen Model

Select a Qwen model based on your GPU memory:

| Model | VRAM Required | Model String |
|-------|---------------|--------------|
| Qwen2.5-0.5B-Instruct | ~2GB | `Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-3B-Instruct | ~8GB | `Qwen/Qwen2.5-3B-Instruct` |
| Qwen2.5-7B-Instruct | ~16GB | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen2.5-14B-Instruct | ~32GB | `Qwen/Qwen2.5-14B-Instruct` |
| Qwen2.5-32B-Instruct | ~65GB | `Qwen/Qwen2.5-32B-Instruct` |

**To use a different model**, edit the `--model` parameter in `docker-compose.yml`:

```yaml
--model Qwen/Qwen2.5-3B-Instruct
```

---

## Step 6: Optional - Configure Hugging Face Token

If you want to use gated models, set up your Hugging Face token:

```bash
# Create .env file
nano .env
```

Add your token:

```
HF_TOKEN=your_huggingface_token_here
```

Update `docker-compose.yml` to include the token in the vLLM service:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  - HF_HOME=/root/.cache/huggingface
  - HF_TOKEN=${HF_TOKEN}
```

---

## Step 7: Start vLLM and Open WebUI Services

```bash
# Pull the Docker images
docker compose pull

# Start both services (first run will download the model)
docker compose up -d

# Watch the logs (this will show model download progress)
docker compose logs -f
```

The first startup will take time as it downloads the model from Hugging Face (several GB depending on the model size).

**Wait for vLLM to be ready** - Look for this message in the logs:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Step 8: Access Open WebUI

### Open Your Browser

Navigate to: **http://localhost:3000**

### First Time Setup

1. **Create an account** - The first user becomes the administrator
2. **Login** with your new credentials
3. The Qwen model should automatically appear in the model selector

### Verify Model Connection

1. Click the **model selector dropdown** at the top of the chat interface
2. You should see **qwen-7b** (or your chosen model) listed
3. Select it to start chatting

---

## Step 9: Test the Setup

### Test from Open WebUI

1. Open http://localhost:3000
2. Select your Qwen model from the dropdown
3. Send a test message: "Hello, who are you?"
4. You should receive a response from the model

### Test vLLM API Directly

**Check service health:**

```bash
curl http://localhost:8000/health
```

**List available models:**

```bash
curl http://localhost:8000/v1/models
```

**Test chat completion:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Monitor Logs

```bash
# Watch vLLM logs
docker compose logs -f vllm

# Watch Open WebUI logs
docker compose logs -f open-webui

# Watch both
docker compose logs -f
```

---

## Step 10: Useful Management Commands

### Stop Services

```bash
docker compose down
```

### Restart Services

```bash
docker compose restart
```

### Restart Individual Service

```bash
docker compose restart vllm
docker compose restart open-webui
```

### View Real-time Logs

```bash
docker compose logs -f vllm
```

### Check Container Status

```bash
docker compose ps
```

### Check Resource Usage

```bash
docker stats
```

### Access Container Shell

```bash
# Access vLLM container
docker compose exec vllm bash

# Access Open WebUI container
docker compose exec open-webui bash
```

### Remove Everything (Including Downloaded Models)

```bash
docker compose down
sudo rm -rf models/ logs/ open-webui/
```

---

## Advanced Configuration Options

### Customize vLLM Parameters

Edit the `command` section in `docker-compose.yml`:

```yaml
command: >
  python -m vllm.entrypoints.openai.api_server
  --model Qwen/Qwen2.5-7B-Instruct
  --host 0.0.0.0
  --port 8000
  --max-model-len 4096
  --gpu-memory-utilization 0.9
  --tensor-parallel-size 1
  --dtype auto
  --trust-remote-code
  --served-model-name qwen-7b
```

**Common Parameters:**
- `--max-model-len`: Maximum sequence length
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--dtype`: Data type (auto, float16, bfloat16)
- `--trust-remote-code`: Required for some models
- `--served-model-name`: Custom name for the model

### Enable Authentication in Open WebUI

Change the Open WebUI environment variable:

```yaml
- WEBUI_AUTH=True  # Changed from False
```

Then restart:

```bash
docker compose restart open-webui
```

### Add Multiple Models

Run multiple vLLM instances with different models:

```yaml
services:
  vllm-qwen-7b:
    # ... 7B model configuration on port 8000
    
  vllm-qwen-3b:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen-3b
    runtime: nvidia
    ports:
      - "8001:8000"
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model Qwen/Qwen2.5-3B-Instruct
      --host 0.0.0.0
      --port 8000
      --served-model-name qwen-3b
    # ... rest of configuration
    networks:
      - ai-network
    
  open-webui:
    environment:
      - OPENAI_API_BASE_URLS=http://vllm-qwen-7b:8000/v1;http://vllm-qwen-3b:8000/v1
      - OPENAI_API_KEYS=sk-dummy-key;sk-dummy-key
```

### Enable RAG and Web Search in Open WebUI

Add to Open WebUI environment:

```yaml
environment:
  - OPENAI_API_BASE_URLS=http://vllm:8000/v1
  - OPENAI_API_KEYS=sk-dummy-key
  - WEBUI_AUTH=False
  - ENABLE_RAG_WEB_SEARCH=True
  - RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  - ENABLE_IMAGE_GENERATION=False
```

---

## Troubleshooting

### GPU Not Detected

**Verify NVIDIA runtime:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check NVIDIA Container Toolkit:**

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Out of Memory Errors

**Solutions:**
- Reduce `--gpu-memory-utilization` to 0.8 or 0.7
- Reduce `--max-model-len` to 2048 or lower
- Choose a smaller model (e.g., 3B instead of 7B)
- Close other GPU-intensive applications

**Check GPU memory usage:**

```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time
```

### Model Not Appearing in Open WebUI

**Check if vLLM is accessible:**

```bash
curl http://localhost:8000/v1/models
```

**Check from within Open WebUI container:**

```bash
docker exec -it open-webui curl http://vllm:8000/v1/models
```

**Verify network connectivity:**

```bash
docker compose ps
docker network inspect vllm-qwen_ai-network
```

### Model Download Fails

**Check internet connection:**

```bash
curl -I https://huggingface.co
```

**Check disk space:**

```bash
df -h
```

**Check Hugging Face token (for gated models):**

```bash
docker compose logs vllm | grep -i token
```

### Open WebUI Can't Connect to vLLM

**Check if both containers are running:**

```bash
docker compose ps
```

**Check vLLM logs for errors:**

```bash
docker compose logs vllm --tail=100
```

**Verify health check:**

```bash
docker compose exec vllm curl http://localhost:8000/health
```

**Check Open WebUI configuration:**

```bash
docker compose exec open-webui env | grep OPENAI
```

### Port Already in Use

**Check what's using the port:**

```bash
sudo lsof -i :8000
sudo lsof -i :3000
```

**Change ports in docker-compose.yml:**

```yaml
ports:
  - "8080:8000"  # Change host port to 8080 for vLLM
  - "3001:8080"  # Change host port to 3001 for Open WebUI
```

### View Detailed Logs

```bash
# View all logs
docker compose logs

# View last 100 lines
docker compose logs --tail=100

# Follow logs in real-time
docker compose logs -f

# View specific service logs
docker compose logs vllm
docker compose logs open-webui
```

---

## Integration with Other Tools

### Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing simply."}
    ]
)

print(response.choices[0].message.content)
```

### cURL Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## Persistent Data Locations

Your data is stored in these directories:

- **vLLM models**: `~/vllm-qwen/models/`
- **Open WebUI data**: `~/vllm-qwen/open-webui/`
- **Logs**: `~/vllm-qwen/logs/`

### Backup Your Data

```bash
cd ~
tar -czf vllm-qwen-backup.tar.gz vllm-qwen/open-webui/
```

### Restore from Backup

```bash
cd ~/vllm-qwen
tar -xzf ~/vllm-qwen-backup.tar.gz
```

---

## Performance Optimization Tips

1. **Adjust GPU Memory Utilization**: Start with 0.9, lower to 0.8 if you experience OOM errors
2. **Use Appropriate Model Size**: Match model size to your GPU VRAM
3. **Enable Tensor Parallelism**: For multi-GPU setups, use `--tensor-parallel-size N`
4. **Tune Max Model Length**: Shorter context = more throughput
5. **Monitor Resource Usage**: Use `nvidia-smi` and `docker stats` regularly

---

## Security Considerations

### Enable Authentication

For production use, always enable authentication:

```yaml
- WEBUI_AUTH=True
```

### Restrict Network Access

Bind to localhost only:

```yaml
ports:
  - "127.0.0.1:3000:8080"  # Only accessible from localhost
  - "127.0.0.1:8000:8000"
```

### Use Reverse Proxy

For external access, use nginx or Traefik with SSL/TLS

---

## Useful Resources

- **vLLM Documentation**: https://docs.vllm.ai
- **Open WebUI Documentation**: https://docs.openwebui.com
- **Qwen Models**: https://huggingface.co/Qwen
- **Docker Documentation**: https://docs.docker.com

---

## Summary

You now have:
- ✅ vLLM running with Qwen model
- ✅ Open WebUI providing a beautiful chat interface
- ✅ OpenAI-compatible API at `http://localhost:8000`
- ✅ Web interface at `http://localhost:3000`
- ✅ GPU acceleration enabled
- ✅ Persistent data storage

Enjoy chatting with your local Qwen model through Open WebUI!