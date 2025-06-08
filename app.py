import os
import io
import base64
import boto3
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from starlette.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional
import traceback
import gc
import json

# Optimize PyTorch for CPU usage
torch.set_num_threads(4)
torch.set_grad_enabled(False)  # Disable gradients for inference

# Force CPU usage
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ----------------------------------------------------------------------
# READ ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def is_s3_configured():
    return all([S3_BUCKET, S3_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY])

# ----------------------------------------------------------------------
# DEFINE GLOBAL VARIABLES
# ----------------------------------------------------------------------

# Use /tmp for temporary/persistent storage
CHECKPOINT_PARENT_DIR = "/tmp"
TMP_CHECKPOINT_DIR = os.path.join(CHECKPOINT_PARENT_DIR, "checkpoint-final")

PIPELINE: Optional[StableDiffusionImg2ImgPipeline] = None
MODEL_LOADING = False
MODEL_LOADED = False
MODEL_LOAD_ERROR = None

FIXED_STRENGTH = 0.5
FIXED_GUIDANCE_SCALE = 7.5
FIXED_STEPS = 20  # Reduced from 30 for faster inference

# ----------------------------------------------------------------------
# FASTAPI APP SETUP
# ----------------------------------------------------------------------

app = FastAPI(
    title="Roomify Interior Img2Img API",
    description="A FastAPI service to run a custom Stable Diffusion Img2Img pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# UTILITY: Persistent storage check
# ----------------------------------------------------------------------

def check_persistent_storage():
    try:
        if not os.path.exists(CHECKPOINT_PARENT_DIR):
            raise RuntimeError(f"Persistent storage base dir {CHECKPOINT_PARENT_DIR} does not exist!")
        test_file = os.path.join(CHECKPOINT_PARENT_DIR, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        with open(test_file, 'r') as f:
            _ = f.read()
        os.remove(test_file)
        print(f"✅ Persistent storage at {CHECKPOINT_PARENT_DIR} is available.")
        return True
    except Exception as e:
        print(f"❌ Persistent storage {CHECKPOINT_PARENT_DIR} check failed: {e}")
        return False

# ----------------------------------------------------------------------
# UTILITY: Download model checkpoint from S3
# ----------------------------------------------------------------------

def download_checkpoint_from_s3():
    if not is_s3_configured():
        raise RuntimeError("S3 configuration incomplete—check all required environment variables!")

    print(f"Starting download from s3://{S3_BUCKET}/{S3_PREFIX}")
    
    # Create a marker file to track download completion
    marker_file = os.path.join(TMP_CHECKPOINT_DIR, ".download_complete")
    
    if os.path.exists(marker_file):
        print("Model already downloaded (marker file exists). Skipping S3 download.")
        return

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    os.makedirs(TMP_CHECKPOINT_DIR, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    except Exception as e:
        raise RuntimeError(f"Error paginating S3 bucket: {e}")

    downloaded_files = 0

    try:
        for page in pages:
            if page is None or page.get("Contents") is None:
                continue
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                relative_path = key[len(S3_PREFIX):]
                local_path = os.path.join(TMP_CHECKPOINT_DIR, relative_path)
                local_dir = os.path.dirname(local_path)
                os.makedirs(local_dir, exist_ok=True)
                
                # Skip if file already exists with same size
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    if local_size == obj['Size']:
                        print(f"Skipping {relative_path} (already exists with correct size)")
                        continue
                
                print(f"Downloading: {relative_path}")
                try:
                    s3.download_file(S3_BUCKET, key, local_path)
                    downloaded_files += 1
                except Exception as inner_e:
                    print(f"Error downloading {key}: {inner_e}")
                    raise
    except Exception as e:
        raise RuntimeError(f"S3 download error: {e}")

    if downloaded_files == 0 and not os.path.exists(os.path.join(TMP_CHECKPOINT_DIR, "model_index.json")):
        raise RuntimeError(f"No files found under s3://{S3_BUCKET}/{S3_PREFIX}")

    # Create marker file
    with open(marker_file, 'w') as f:
        f.write(f"Downloaded {downloaded_files} files")
        
    print(f"✅ Downloaded {downloaded_files} files to {TMP_CHECKPOINT_DIR}")

def fix_missing_model_components():
    model_index_path = os.path.join(TMP_CHECKPOINT_DIR, "model_index.json")
    if not os.path.exists(model_index_path):
        print("Creating model_index.json...")
        model_index = {
            "_class_name": "StableDiffusionImg2ImgPipeline",
            "_diffusers_version": "0.21.0",
            "feature_extractor": ["transformers", "CLIPImageProcessor"],
            "requires_safety_checker": False,
            "safety_checker": None,
            "scheduler": ["diffusers", "PNDMScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
        with open(model_index_path, 'w') as f:
            json.dump(model_index, f, indent=2)

    # Create scheduler config if missing
    scheduler_dir = os.path.join(TMP_CHECKPOINT_DIR, "scheduler")
    if not os.path.exists(scheduler_dir):
        os.makedirs(scheduler_dir, exist_ok=True)
        scheduler_config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.21.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
        with open(os.path.join(scheduler_dir, "scheduler_config.json"), 'w') as f:
            json.dump(scheduler_config, f, indent=2)

def verify_model_files():
    """Verify that critical model files exist"""
    critical_files = [
        "model_index.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/config.json",
        "text_encoder/pytorch_model.bin",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "vae/diffusion_pytorch_model.bin",
        "vae/config.json"
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = os.path.join(TMP_CHECKPOINT_DIR, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ Missing critical files: {missing_files}")
        return False
    
    print("✅ All critical model files verified")
    return True

# ----------------------------------------------------------------------
# ASYNC MODEL LOADING
# ----------------------------------------------------------------------

async def load_model_async():
    global PIPELINE, MODEL_LOADING, MODEL_LOADED, MODEL_LOAD_ERROR

    MODEL_LOADING = True
    try:
        # Clear any existing model from memory
        if PIPELINE is not None:
            del PIPELINE
            gc.collect()
        
        if not os.path.exists(os.path.join(TMP_CHECKPOINT_DIR, "model_index.json")):
            print("Model not found locally. Downloading from S3...")
            check_persistent_storage()
            download_checkpoint_from_s3()
            fix_missing_model_components()
        else:
            print("Model found locally. Skipping download.")

        # Verify files before loading
        if not verify_model_files():
            raise RuntimeError("Model files verification failed")

        print("Loading Stable Diffusion pipeline...")
        
        # Load with CPU optimizations
        PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
            TMP_CHECKPOINT_DIR,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
            device_map=None,
            local_files_only=True,
        )
        
        # Use memory-efficient scheduler
        PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config)
        
        # Move to CPU explicitly
        PIPELINE = PIPELINE.to(DEVICE)
        
        # Enable memory efficient attention if available
        if hasattr(PIPELINE.unet, 'set_attention_processor'):
            from diffusers.models.attention_processor import AttnProcessor
            PIPELINE.unet.set_attention_processor(AttnProcessor())
        
        # ⚠️ DO NOT use enable_sequential_cpu_offload() on CPU-only
        # if hasattr(PIPELINE, 'enable_sequential_cpu_offload'):
        #     PIPELINE.enable_sequential_cpu_offload()

        MODEL_LOADED = True
        MODEL_LOADING = False
        MODEL_LOAD_ERROR = None

        gc.collect()
        print("🚀 Model loaded successfully!")

    except Exception as e:
        MODEL_LOADING = False
        MODEL_LOADED = False
        tb_str = traceback.format_exc()
        MODEL_LOAD_ERROR = f"{e}\n{tb_str}"
        print(f"❌ Failed to load model: {MODEL_LOAD_ERROR}")
        gc.collect()

# ----------------------------------------------------------------------
# STARTUP EVENT
# ----------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    print("FastAPI starting up...")
    print(f"Device: {DEVICE}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"S3 Prefix: {S3_PREFIX}")

    if not check_persistent_storage():
        print(f"WARNING: Persistent storage {CHECKPOINT_PARENT_DIR} is not available or not writable.")
    elif not is_s3_configured():
        print("WARNING: S3 credentials/configuration are incomplete. Model loading will fail.")
    else:
        asyncio.create_task(load_model_async())
        print("Model loading started in background...")

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.get("/", summary="Simple healthcheck")
def read_root():
    return {
        "message": "Roomify Img2Img API is up and running!",
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "model_error": MODEL_LOAD_ERROR,
        "device": str(DEVICE)
    }

@app.get("/health")
def health_check():
    if MODEL_LOAD_ERROR:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": MODEL_LOAD_ERROR},
        )
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING
    }

@app.get("/model-status")
def model_status():
    return {
        "loaded": MODEL_LOADED,
        "loading": MODEL_LOADING,
        "error": MODEL_LOAD_ERROR,
        "device": str(DEVICE),
        "checkpoint_dir": TMP_CHECKPOINT_DIR,
        "checkpoint_exists": os.path.exists(TMP_CHECKPOINT_DIR),
    }

@app.post("/generate", summary="Generate a new image from an existing room photo + prompt")
async def generate_image(
    prompt: str = Form(..., description="The text prompt (must include your special token, e.g. `<interiorx>`)."),
    image: UploadFile = File(..., description="A JPEG/PNG room image to transform"),
):
    if MODEL_LOADING:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please try again in a few moments.",
            headers={"Retry-After": "30"}
        )
    if MODEL_LOAD_ERROR:
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {MODEL_LOAD_ERROR}"
        )
    if not MODEL_LOADED or PIPELINE is None:
        if not MODEL_LOADING and is_s3_configured() and check_persistent_storage():
            asyncio.create_task(load_model_async())
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Loading has been triggered. Please try again in a few moments.",
            headers={"Retry-After": "60"}
        )
    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPEG/PNG.")

    try:
        pipe_call = PIPELINE.__call__
        # Since we're CPU-only, do not use autocast (which is CUDA-specific)
        with torch.no_grad():
            output = pipe_call(
                prompt=prompt,
                image=input_image,
                strength=FIXED_STRENGTH,
                guidance_scale=FIXED_GUIDANCE_SCALE,
                num_inference_steps=FIXED_STEPS
            )
        result_image = output.images[0]
        gc.collect()
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Img2Img generation failed: {e}")

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# For running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
