#!/bin/bash
set -e

echo "Starting Wan 2.1 Quantized Model Setup..."

# Activate Python environment
source /venv/main/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install fastapi uvicorn python-multipart torch==2.0.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.23.5  # Compatible with both ComfyUI and TTS

# Create project directories
echo "Creating project directories..."
mkdir -p /workspace/models
cd /workspace

# Set up ComfyUI
echo "Setting up ComfyUI..."
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt

# Install GGUF support
echo "Installing GGUF support..."
mkdir -p custom_nodes
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
pip install --upgrade gguf
cd ..

# Create model directories
echo "Creating model directories..."
mkdir -p models/unet models/text_encoder models/vae outputs tts_outputs

# Download models
echo "Downloading quantized Wan 2.1 models (this may take some time)..."
wget https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/Wan2.1-T2V-14B-q5_k.gguf -O models/unet/Wan2.1-T2V-14B-q5_k.gguf
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/text_encoder/pytorch_model.bin -O models/text_encoder/umt5-xxl-enc-bf16.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/vae/pytorch_model.bin -O models/vae/vae-ft-mse-840000-ema-pruned.safetensors

# Create systemd service for ComfyUI web UI
cat > /etc/systemd/system/comfyui.service << 'EOFS'
[Unit]
Description=ComfyUI Web Interface
After=network.target

[Service]
User=root
WorkingDirectory=/workspace/ComfyUI
ExecStart=/venv/main/bin/python main.py --listen 0.0.0.0 --port 8188
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOFS

# Enable and start the service
systemctl enable comfyui
systemctl start comfyui

echo "Setup complete! ComfyUI is running on port 8188"
