# Wan 2.1 Quantized Model Setup

Setup script for deploying quantized Wan 2.1 video generation model with ComfyUI interface on Vast.ai.

## Features

- Quantized Wan 2.1 14B model for efficient video generation
- ComfyUI web interface for workflow creation and testing
- Systemd service for reliable operation
- Compatible PyTorch and NumPy versions

## Usage with Vast.ai

### Creating an Instance

Use this command to create a Vast.ai instance with all necessary configuration:

```bash
vastai create instance <OFFER_ID> --image vastai/pytorch:2.5.1-cuda-12.1.1 --env '-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 8000:8000 -p 8188:8188 -p 22:22 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8188:8188:/:ComfyUI|localhost:6006:16006:/:Tensorboard" -e PROVISIONING_SCRIPT="https://raw.githubusercontent.com/altommo/wan21-quantized-setup/main/setup.sh" -e AUTH_EXCLUDE="8000,8188"' --disk 500 --jupyter --ssh --direct
```

Find an appropriate `<OFFER_ID>` using:
```bash
vastai search offers --disk 50 --gpu-ram 80 --gpu-name A100
```

### SSH Setup

1. Generate an SSH key:
   ```bash
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/vast_key
   ```

2. Add the key to Vast.ai:
   ```bash
   vastai create ssh-key "$(cat ~/.ssh/vast_key.pub)"
   ```

3. Add the key to your instance via the Vast.ai web interface

4. Connect to your instance:
   ```bash
   ssh -i ~/.ssh/vast_key root@YOUR_IP -p YOUR_PORT
   ```

### Access ComfyUI Web Interface

1. Via SSH tunnel:
   ```bash
   ssh -i ~/.ssh/vast_key -L 8188:localhost:8188 root@YOUR_IP -p YOUR_PORT
   ```
   Then access http://localhost:8188 in your browser

2. Or directly via external URL:
   ```
   http://YOUR_IP:8188
   ```

## Working with the Wan 2.1 Model

1. In ComfyUI, load the workflow:
   - Create a UnetLoaderGGUF node
   - Set the model path to: `models/unet/Wan2.1-T2V-14B-q5_k.gguf`
   - Create CLIPTextEncode for your prompt
   - Connect to VideoGenPipeline and WriteVideo nodes

2. Recommended settings:
   - Frames: 8-16 for testing (more for production)
   - Resolution: 384x224 for quick tests, up to 512x320 for better quality
   - Steps: 30-50 (more steps = better quality, longer generation)

## Resource Requirements

- **VRAM**: ~35-40GB for video generation
- **Storage**: 20GB+ for models, more for generated content
- **Recommended GPU**: NVIDIA A100 (80GB)

## License

This project is provided under the MIT License.
