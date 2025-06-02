import os
import shutil
import boto3
import io

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# Environment variables (configure these in Render’s Environment settings)
# ──────────────────────────────────────────────────────────────────────────────
S3_BUCKET             = os.getenv("S3_BUCKET")               # e.g. "my-roomify-models"
S3_PREFIX             = os.getenv("S3_PREFIX")               # e.g. "roomify-checkpoint-final/"
AWS_REGION            = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Inference parameters (hard‐coded)
FIXED_STRENGTH   = float(os.getenv("FIXED_STRENGTH", "0.75"))
FIXED_GUIDANCE   = float(os.getenv("FIXED_GUIDANCE", "7.5"))
FIXED_STEPS      = int(os.getenv("FIXED_STEPS", "30"))

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This global will hold our loaded pipeline
pipeline = None


def download_checkpoint_from_s3():
    """
    Download all objects under S3_BUCKET/S3_PREFIX into ./checkpoint-final/,
    preserving folder structure. Skip any “folder placeholder” (keys ending in '/').
    """
    if not S3_BUCKET or not S3_PREFIX:
        raise RuntimeError("S3_BUCKET and S3_PREFIX must both be set in the environment.")

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Instead of /tmp, write to a folder in the app’s working directory:
    local_root = os.path.join(os.getcwd(), "checkpoint-final")

    # If it already exists (from previous run), delete it:
    if os.path.exists(local_root):
        shutil.rmtree(local_root)
    os.makedirs(local_root, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. "roomify-checkpoint-final/unet/config.json"

            # Skip "folder" keys (they end with "/")
            if key.endswith("/"):
                continue

            # Compute path relative to the prefix:
            # If S3_PREFIX="roomify-checkpoint-final/"
            # and key="roomify-checkpoint-final/unet/config.json",
            # then rel_path = "unet/config.json"
            rel_path = os.path.relpath(key, S3_PREFIX)

            # Full local path
            local_file_path = os.path.join(local_root, rel_path)

            # Ensure subfolders exist
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download into that exact filename
            s3.download_file(S3_BUCKET, key, local_file_path)
            print(f"Downloaded s3://{S3_BUCKET}/{key} → {local_file_path}")

    print(f"✅ Finished downloading checkpoint into {local_root}")


def load_img2img_pipeline():
    """
    1) Downloads the fine‐tuned pieces from S3 into ./checkpoint-final/.
    2) Instantiates a base Img2Img pipeline from the cache (v1.5), then overwrites
       its tokenizer, text_encoder, and unet with our fine‐tuned folders.
    3) Replaces the scheduler with UniPCMultistepScheduler.
    4) Moves pipeline onto DEVICE.
    """
    global pipeline

    # 1) Download your fine‐tuned checkpoints into ./checkpoint-final/
    download_checkpoint_from_s3()
    local_root = os.path.join(os.getcwd(), "checkpoint-final")

    # 2) Load a base Img2Img pipeline from the Hugging Face cache (not from S3),
    #    assuming you already cached v1.5 during build time.
    base_model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )

    # 3) Overwrite tokenizer + text_encoder with our fine‐tuned versions:
    pipe.tokenizer    = CLIPTokenizer.from_pretrained(os.path.join(local_root, "tokenizer"))
    pipe.text_encoder = CLIPTextModel.from_pretrained(os.path.join(local_root, "text_encoder"))

    # 4) Overwrite UNet with our fine‐tuned UNet:
    pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(local_root, "unet"))

    # (Optional) If you also fine‐tuned VAE, you could do:
    # pipe.vae = AutoencoderKL.from_pretrained(os.path.join(local_root, "vae"))

    # 5) Swap in UniPCMultistepScheduler for faster sampling:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 6) Move pipeline onto GPU (if available)
    pipe = pipe.to(DEVICE)

    pipeline = pipe
    print("✅ Img2Img pipeline loaded and ready.")


@app.on_event("startup")
def on_startup():
    """
    At startup:
      • Download S3 checkpoint → ./checkpoint-final/
      • Load/overwrite the pipeline components
    """
    try:
        load_img2img_pipeline()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Img2Img pipeline: {e}")


@app.get("/health")
def health_check():
    """
    Simple healthcheck returning status and inference settings.
    """
    if pipeline is None:
        return {"status": "error", "detail": "Pipeline not loaded."}
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_dir": os.path.join(os.getcwd(), "checkpoint-final"),
        "fixed_strength": FIXED_STRENGTH,
        "fixed_guidance": FIXED_GUIDANCE,
        "fixed_steps": FIXED_STEPS,
    }


@app.post("/generate")
async def generate_image(prompt: str, image: UploadFile = File(...)):
    """
    Expects multipart/form-data with:
      • prompt: a string containing "<interiorx>"
      • image: the room photo (JPEG/PNG)
    Returns a generated image as raw PNG.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    # Read the uploaded image into a PIL Image and resize to 512×512
    try:
        contents = await image.read()
        init_image = Image.open(io.BytesIO(contents)).convert("RGB")
        init_image = init_image.resize((512, 512), resample=Image.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Run Img2Img with fixed parameters
    try:
        results = pipeline(
            prompt=prompt,
            image=init_image,
            strength=FIXED_STRENGTH,
            guidance_scale=FIXED_GUIDANCE,
            num_inference_steps=FIXED_STEPS,
        )
        generated = results.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Convert to in‐memory PNG and stream back
    buffer = io.BytesIO()
    generated.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
