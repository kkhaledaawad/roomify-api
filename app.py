import os
import io
import shutil
import boto3

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# Environment variables (set these in Render’s Environment settings)
# ──────────────────────────────────────────────────────────────────────────────
S3_BUCKET            = os.getenv("S3_BUCKET")
S3_PREFIX            = os.getenv("S3_PREFIX")
AWS_REGION           = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
AWS_ACCESS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY")

# Inference parameters (hard‐coded; no need to pass them in every request)
FIXED_STRENGTH   = float(os.getenv("FIXED_STRENGTH", "0.75"))
FIXED_GUIDANCE   = float(os.getenv("FIXED_GUIDANCE", "7.5"))
FIXED_STEPS      = int(os.getenv("FIXED_STEPS", "30"))

# Device (will be "cuda" if a GPU is available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder for our loaded Img2Img pipeline
pipeline = None


def download_checkpoint_from_s3():
    """
    Download all files under S3_BUCKET/S3_PREFIX into /tmp/checkpoint-final/
    while preserving the folder structure.

    Expects:
      • S3_BUCKET (e.g. "my-roomify-models")
      • S3_PREFIX (e.g. "roomify-checkpoint-final/")
      • AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY set in the environment
      • AWS_REGION set to the region of your bucket
    """
    if not S3_BUCKET or not S3_PREFIX:
        raise RuntimeError("S3_BUCKET and S3_PREFIX must be set as environment variables.")

    # Create an S3 client using the credentials from environment variables:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Local root directory where we will mirror the S3 prefix
    local_root = "/tmp/checkpoint-final"

    # If it already exists from a prior run, remove it first:
    if os.path.exists(local_root):
        shutil.rmtree(local_root)
    os.makedirs(local_root, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    # Paginate through all objects under the given prefix
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. "roomify-checkpoint-final/unet/config.json"

            # Skip any "folder placeholders" (keys that end with '/')
            if key.endswith("/"):
                continue

            # Determine the path relative to S3_PREFIX
            # e.g. if S3_PREFIX = "roomify-checkpoint-final/"
            #      and key = "roomify-checkpoint-final/unet/config.json",
            # then rel_path = "unet/config.json"
            rel_path = os.path.relpath(key, S3_PREFIX)

            # Compute the full local file path
            local_file_path = os.path.join(local_root, rel_path)

            # Make sure any intermediate directories exist
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download the file from S3 into the exact local path
            s3.download_file(S3_BUCKET, key, local_file_path)
            print(f"Downloaded s3://{S3_BUCKET}/{key} → {local_file_path}")

    print(f"✅ Finished downloading checkpoint into {local_root}")


def load_img2img_pipeline():
    """
    Called on application startup. Downloads the checkpoint from S3
    and loads the StableDiffusionImg2ImgPipeline into the global `pipeline`.
    """
    global pipeline

    # Step 1: Download from S3 to /tmp/checkpoint-final/
    download_checkpoint_from_s3()
    local_checkpoint_dir = "/tmp/checkpoint-final"

    # Step 2: Load the Img2Img pipeline from that directory
    # We assume the directory structure matches a Diffusers checkpoint:
    #   ├── text_encoder/
    #   ├── tokenizer/
    #   └── unet/
    #
    # In Diffusers v0.14.0 or later, you can do:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        local_checkpoint_dir,
        safety_checker=None, 
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
    )

    # Step 3: Replace the scheduler with UniPCMultistepScheduler for faster sampling
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Step 4: Move the pipeline to the GPU (if available) or keep on CPU otherwise
    pipe = pipe.to(DEVICE)

    pipeline = pipe
    print("✅ Img2Img pipeline loaded and ready.")


@app.on_event("startup")
def on_startup():
    """
    FastAPI startup event: download checkpoint + load pipeline.
    """
    try:
        load_img2img_pipeline()
    except Exception as e:
        # If anything goes wrong at startup, re‐raise so the service fails
        raise RuntimeError(f"Failed to initialize Img2Img pipeline: {e}")


@app.get("/health")
def health_check():
    """
    Health endpoint to verify that the service is up and the model is loaded.
    """
    if pipeline is None:
        return {"status": "error", "detail": "Pipeline not loaded."}

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_dir": "/tmp/checkpoint-final",
        "fixed_strength": FIXED_STRENGTH,
        "fixed_guidance": FIXED_GUIDANCE,
        "fixed_steps": FIXED_STEPS
    }


@app.post("/generate")
async def generate_image(prompt: str, image: UploadFile = File(...)):
    """
    Accepts a multipart/form-data request containing:
      • prompt (string, must include "<interiorx>")
      • image (the input room photo as a JPEG/PNG upload)
    
    Returns a generated image (PNG) as a StreamingResponse.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    # Read the uploaded image into a PIL Image
    try:
        contents = await image.read()
        init_image = Image.open(io.BytesIO(contents)).convert("RGB")
        init_image = init_image.resize((512, 512), resample=Image.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Run the Img2Img pipeline with fixed parameters
    try:
        results = pipeline(
            prompt=prompt,
            image=init_image,
            strength=FIXED_STRENGTH,
            guidance_scale=FIXED_GUIDANCE,
            num_inference_steps=FIXED_STEPS
        )
        generated = results.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Convert the PIL image to PNG bytes in memory
    buffer = io.BytesIO()
    generated.save(buffer, format="PNG")
    buffer.seek(0)

    # Return the PNG as a streaming response
    return StreamingResponse(buffer, media_type="image/png")
