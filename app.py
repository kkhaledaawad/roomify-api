import os
import io
import base64
import boto3
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from starlette.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional

# ----------------------------------------------------------------------
# READ ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")  # e.g., "roomify-checkpoint-final/"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Check environment variables only if we're in production
# For local testing, you might want to skip this
if os.getenv("WEBSITE_SITE_NAME"):  # This env var exists in Azure App Service
    if not all([S3_BUCKET, S3_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("WARNING: S3 credentials not fully configured. Model loading will fail.")

# ----------------------------------------------------------------------
# DEFINE GLOBAL VARIABLES
# ----------------------------------------------------------------------

TMP_CHECKPOINT_DIR = "/tmp/checkpoint-final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE: Optional[StableDiffusionImg2ImgPipeline] = None
MODEL_LOADING = False
MODEL_LOADED = False
MODEL_LOAD_ERROR = None

# You can fix these hyperparameters as constants:
FIXED_STRENGTH = 0.5       # how much to preserve the original image (0.0–1.0)
FIXED_GUIDANCE_SCALE = 7.5 # standard guidance scale
FIXED_STEPS = 30           # number of denoising steps

# ----------------------------------------------------------------------
# FASTAPI APP SETUP
# ----------------------------------------------------------------------

app = FastAPI(
    title="Roomify Interior Img2Img API",
    description="A FastAPI service to run a custom Stable Diffusion Img2Img pipeline.",
    version="1.0.0",
)

# Optionally enable CORS if your frontend is hosted elsewhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# UTILITY: Download model checkpoint from S3 into /tmp/checkpoint-final
# ----------------------------------------------------------------------

def download_checkpoint_from_s3():
    """
    Lists all objects under S3_PREFIX in S3_BUCKET, creates matching local directories
    under TMP_CHECKPOINT_DIR, and downloads every file. Preserves folder structure.
    """
    print(f"Starting download from s3://{S3_BUCKET}/{S3_PREFIX}")
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )

    # Ensure base directory exists
    os.makedirs(TMP_CHECKPOINT_DIR, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    downloaded_any = False
    total_files = 0
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip if it's the prefix folder itself
            if key.endswith("/"):
                continue

            # Compute relative path inside TMP_CHECKPOINT_DIR
            relative_path = key[len(S3_PREFIX):]
            local_path = os.path.join(TMP_CHECKPOINT_DIR, relative_path)

            # Make sure local directory exists
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download
            print(f"Downloading: {relative_path}")
            s3.download_file(S3_BUCKET, key, local_path)
            downloaded_any = True
            total_files += 1

    if not downloaded_any:
        raise RuntimeError(f"No files found under s3://{S3_BUCKET}/{S3_PREFIX}")
    
    print(f"✅ Downloaded {total_files} files to {TMP_CHECKPOINT_DIR}")

# ----------------------------------------------------------------------
# ASYNC MODEL LOADING
# ----------------------------------------------------------------------

async def load_model_async():
    """Load the model asynchronously after startup"""
    global PIPELINE, MODEL_LOADING, MODEL_LOADED, MODEL_LOAD_ERROR
    
    MODEL_LOADING = True
    try:
        # Check if model already exists locally (for faster restarts)
        if not os.path.exists(os.path.join(TMP_CHECKPOINT_DIR, "model_index.json")):
            print("Model not found locally. Downloading from S3...")
            download_checkpoint_from_s3()
        else:
            print("Model found locally. Skipping download.")
        
        print("Loading Stable Diffusion pipeline...")
        PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
            TMP_CHECKPOINT_DIR,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True,  # Important for Azure
        )

        # Replace the default scheduler with UniPC Multistep
        PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config)

        # Move pipeline to device
        PIPELINE.to(DEVICE)

        MODEL_LOADED = True
        MODEL_LOADING = False
        print("🚀 Model loaded successfully!")
        
    except Exception as e:
        MODEL_LOADING = False
        MODEL_LOAD_ERROR = str(e)
        print(f"❌ Failed to load model: {e}")

# ----------------------------------------------------------------------
# STARTUP EVENT - Quick startup, load model in background
# ----------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    print("FastAPI starting up...")
    print(f"Device: {DEVICE}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"S3 Prefix: {S3_PREFIX}")
    
    # Start model loading in the background
    asyncio.create_task(load_model_async())
    print("Model loading started in background...")

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.get("/", summary="Simple healthcheck")
def read_root():
    return {
        "message": "Roomify Img2Img API is up and running!",
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "model_error": MODEL_LOAD_ERROR,
        "device": str(DEVICE)
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Azure"""
    if MODEL_LOAD_ERROR:
        return {"status": "unhealthy", "error": MODEL_LOAD_ERROR}, 503
    
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING
    }

@app.get("/model-status")
def model_status():
    """Check model loading status"""
    return {
        "loaded": MODEL_LOADED,
        "loading": MODEL_LOADING,
        "error": MODEL_LOAD_ERROR,
        "device": str(DEVICE),
        "checkpoint_dir": TMP_CHECKPOINT_DIR,
        "checkpoint_exists": os.path.exists(TMP_CHECKPOINT_DIR)
    }

@app.post("/generate", summary="Generate a new image from an existing room photo + prompt")
async def generate_image(
    prompt: str = Form(..., description="The text prompt (must include your special token, e.g. `<interiorx>`)."),
    image: UploadFile = File(..., description="A JPEG/PNG room image to transform"),
):
    """
    Accepts:
      - prompt: string (e.g. "<interiorx> a cozy Scandinavian living room")
      - image: multipart form file (JPEG/PNG)
    Returns:
      - PNG image bytes as StreamingResponse
    """
    # Check if model is loaded
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
        # Try to load model if not already loading
        if not MODEL_LOADING:
            asyncio.create_task(load_model_async())
        raise HTTPException(
            status_code=503, 
            detail="Model not ready. Loading has been triggered. Please try again in a few moments.",
            headers={"Retry-After": "60"}
        )

    # 1) Read and preprocess the uploaded image
    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Diffusers expects 512×512 for img2img
        input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPEG/PNG.")

    # 2) Run the img2img pipeline with fixed strength, guidance, and steps
    try:
        with torch.autocast("cuda") if DEVICE.type == "cuda" else torch.no_grad():
            output = PIPELINE(
                prompt=prompt,
                image=input_image,
                strength=FIXED_STRENGTH,
                guidance_scale=FIXED_GUIDANCE_SCALE,
                num_inference_steps=FIXED_STEPS,
            )
        result_image = output.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Img2Img generation failed: {e}")

    # 3) Encode output image to PNG bytes
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# For running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
