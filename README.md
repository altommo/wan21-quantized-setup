# Wan 2.1 Quantized Model Setup

Setup script for deploying quantized Wan 2.1 video generation model with voice cloning TTS capabilities on Vast.ai.

## Features

- Quantized Wan 2.1 14B model for efficient video generation
- XTTS v2 voice cloning for documentary-style narration
- FastAPI endpoints for easy integration
- Supervisor-managed services for reliability

## Usage with Vast.ai

### Creating an Instance

Use this command to create a Vast.ai instance with all necessary configuration:

```bash
vastai create instance <OFFER_ID> --image vastai/pytorch:2.5.1-cuda-12.1.1 --env '-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 8000:8000 -p 8001:8001 -p 22:22 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8000:8000:/:Wan2.1 API|localhost:8001:8001:/:TTS API|localhost:6006:16006:/:Tensorboard" -e PROVISIONING_SCRIPT="https://raw.githubusercontent.com/altommo/wan21-quantized-setup/main/setup.sh" -e AUTH_EXCLUDE="8000,8001"' --disk 500 --jupyter --ssh --direct
```

Find an appropriate `<OFFER_ID>` using:
```bash
vastai search offers --disk 50 --gpu-ram 80 --gpu-name A100
```

### API Endpoints

#### Video Generation API

```bash
# Generate a video
curl -X POST "http://your-server-ip:8000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=a beautiful sunset over mountains" \
  -F "frames=16" \
  -F "width=512" \
  -F "height=320" \
  -F "fps=8"

# Check generation status and download
curl -X GET "http://your-server-ip:8000/result/JOB_ID_HERE" --output video.mp4
```

#### Voice Cloning API

```bash
# Clone a voice
curl -X POST "http://your-server-ip:8001/clone_voice" \
  -H "Content-Type: multipart/form-data" \
  -F "text=This is a test of the voice cloning system for documentary narration." \
  -F "reference_audio=@your_voice_sample.wav" \
  -F "language=en"

# Download generated audio
curl -X GET "http://your-server-ip:8001/voice/JOB_ID_HERE" --output audio.wav
```

## Local Access via SSH Tunnel

For secure access, use SSH tunneling:

```bash
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 root@your-server-ip -p your-ssh-port
```

Then access the APIs locally:
- http://localhost:8000/healthcheck
- http://localhost:8001/clone_voice

## Resource Requirements

- **VRAM**: ~35-40GB for video generation
- **Storage**: 20GB+ for models, more for generated content
- **Recommended GPU**: NVIDIA A100 (80GB)

## License

This project is provided under the MIT License.
