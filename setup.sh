#!/bin/bash
set -e

echo "Starting Wan 2.1 Quantized Model Setup..."

# Activate Python environment
source /venv/main/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install fastapi uvicorn python-multipart torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install realesrgan-ncnn-vulkan ffmpeg-python

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

# Set up TTS
echo "Setting up TTS..."
cd /workspace
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .

# Download TTS model
echo "Downloading TTS model..."
python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')"

# Create API scripts
echo "Creating API scripts..."
cd /workspace/ComfyUI

# Wan 2.1 API
cat > api.py << 'EOF'
import os
import uuid
import json
import asyncio
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Initialize FastAPI
app = FastAPI(title="Wan2.1 GGUF Video Generation API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load workflow template
WORKFLOW_TEMPLATE = {
  "prompt": {
    "3": {
      "inputs": {
        "seed": 42,
        "steps": 50
      },
      "class_type": "SeedGenerator"
    },
    "4": {
      "inputs": {
        "text": "a cute cat playing with a ball",
        "max_length": 128
      },
      "class_type": "CLIPTextEncode"
    },
    "6": {
      "inputs": {
        "model_path": "models/unet/Wan2.1-T2V-14B-q5_k.gguf"
      },
      "class_type": "UnetLoaderGGUF"
    },
    "7": {
      "inputs": {
        "samples": ["4", 0],
        "seed": ["3", 0],
        "unet": ["6", 0],
        "frames": 16,
        "width": 512,
        "height": 320,
        "cfg": 7.5
      },
      "class_type": "VideoGenPipeline"
    },
    "8": {
      "inputs": {
        "video": ["7", 0],
        "fps": 8,
        "format": "mp4",
        "filename_prefix": "output"
      },
      "class_type": "WriteVideo"
    }
  }
}

# Save workflow template
with open("workflow.json", "w") as f:
    json.dump(WORKFLOW_TEMPLATE, f)

# Generation function
async def generate_video(prompt, output_path, seed=None, width=512, height=320, frames=16, fps=8):
    # Set up ComfyUI execution params
    prompt_data = WORKFLOW_TEMPLATE.copy()
    
    # Update workflow with parameters
    prompt_data["prompt"]["4"]["inputs"]["text"] = prompt
    
    if seed is not None:
        prompt_data["prompt"]["3"]["inputs"]["seed"] = seed
    
    # Update resolution and frames
    prompt_data["prompt"]["7"]["inputs"]["width"] = width
    prompt_data["prompt"]["7"]["inputs"]["height"] = height
    prompt_data["prompt"]["7"]["inputs"]["frames"] = frames
    prompt_data["prompt"]["8"]["inputs"]["fps"] = fps
    
    # Save workflow file
    execution_id = str(uuid.uuid4())
    workflow_path = f"/tmp/workflow_{execution_id}.json"
    with open(workflow_path, "w") as f:
        json.dump(prompt_data, f)
    
    # Execute ComfyUI in headless mode
    cmd = f"python main.py --gpu-only --port 8188 --input {workflow_path} --output {output_path} --disable-metadata"
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    print(f"Generation complete. Output: {stdout.decode()}")
    if stderr:
        print(f"Errors: {stderr.decode()}")
    
    # Clean up temporary files
    if os.path.exists(workflow_path):
        os.remove(workflow_path)
    
    return output_path

@app.post("/generate")
async def create_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(320),
    frames: Optional[int] = Form(16),
    fps: Optional[int] = Form(8)
):
    # Create unique output path
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_id = str(uuid.uuid4())
    output_path = f"{output_dir}/{output_id}.mp4"
    
    # Generate video in background
    background_tasks.add_task(
        generate_video, 
        prompt, 
        output_path, 
        seed,
        width,
        height,
        frames,
        fps
    )
    
    return {
        "message": "Video generation started",
        "job_id": output_id,
        "estimated_time_minutes": (frames * 50) / 380  # Rough estimate
    }

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    file_path = f"outputs/{job_id}.mp4"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"status": "processing", "message": "Video still generating or job ID not found"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy", "model": "Wan2.1-T2V-14B-gguf"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
EOF

# TTS API
cat > tts_api.py << 'EOF'
import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI(title="Voice Cloning TTS API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models lazily
tts_model = None
config = None

def get_tts_model():
    global tts_model, config
    if tts_model is None:
        config = XttsConfig()
        config.load_json("/workspace/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
        tts_model = Xtts.init_from_config(config)
        tts_model.load_checkpoint(config, checkpoint_dir="/workspace/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
        tts_model.to("cuda")
    return tts_model, config

async def generate_speech(text, reference_file, output_path, language="en"):
    model, config = get_tts_model()
    
    # Generate speech
    outputs = model.synthesize(
        text,
        config.audio.sample_rate,
        reference_file,
        reference_speaker_name=None,
        language=language,
        temperature=0.7,
    )
    
    # Save audio file
    model.save_wav(outputs["wav"], output_path)
    return output_path

@app.post("/clone_voice")
async def clone_voice(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    # Save reference audio
    output_dir = "tts_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    ref_file = f"{output_dir}/reference_{uuid.uuid4()}.wav"
    with open(ref_file, "wb") as f:
        f.write(await reference_audio.read())
    
    # Generate unique output path
    output_id = str(uuid.uuid4())
    output_path = f"{output_dir}/{output_id}.wav"
    
    # Generate speech in background
    background_tasks.add_task(
        generate_speech,
        text,
        ref_file,
        output_path,
        language
    )
    
    return {
        "message": "Voice generation started",
        "job_id": output_id,
        "estimated_time_seconds": len(text.split()) * 0.15  # Rough estimate
    }

@app.get("/voice/{job_id}")
async def get_voice(job_id: str):
    file_path = f"tts_outputs/{job_id}.wav"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"status": "processing", "message": "Voice still generating or job ID not found"}

if __name__ == "__main__":
    uvicorn.run("tts_api:app", host="0.0.0.0", port=8001)
EOF

# Start APIs
echo "Starting API servers..."
# Start APIs using supervisor for auto-restart and logging
cat > /etc/supervisor/conf.d/wan-api.conf << EOF
[program:wan-api]
command=/venv/main/bin/python /workspace/ComfyUI/api.py
directory=/workspace/ComfyUI
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/wan-api.err.log
stdout_logfile=/var/log/supervisor/wan-api.out.log
EOF

cat > /etc/supervisor/conf.d/tts-api.conf << EOF
[program:tts-api]
command=/venv/main/bin/python /workspace/ComfyUI/tts_api.py
directory=/workspace/ComfyUI
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/tts-api.err.log
stdout_logfile=/var/log/supervisor/tts-api.out.log
EOF

# Restart supervisor to load the new configs
supervisorctl update

# Add helpful message to motd
cat > /etc/motd << EOF
==========================================================
Wan 2.1 Quantized Model & TTS Setup Complete!

APIs available at:
- Video Generation: http://localhost:8000
- Voice Cloning TTS: http://localhost:8001

Example usage:
- Generate video: 
  curl -X POST "http://localhost:8000/generate" -F "prompt=beautiful sunset" -F "frames=16"

- Clone voice:
  curl -X POST "http://localhost:8001/clone_voice" -F "text=Hello world" -F "reference_audio=@voice.wav"

Documentation available at: https://github.com/altommo/wan21-quantized-setup
==========================================================
EOF

echo "Setup complete! APIs are running on ports 8000 and 8001"
