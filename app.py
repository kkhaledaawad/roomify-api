import os
import shutil
import boto3

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
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

# Hard‐coded inference parameters (no need to pass each time)
FIXED_STRENGTH   = float(os.getenv("FIXED_STRENGTH", "0.75"))
FIXED_GUIDANCE   = float(os.getenv("FIXED_GUIDANCE", "7.5"))
FIXED_STEPS      = int(os.getenv("FIXED_STEPS", "30"))

# Which device to run on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This will hold our loaded pipeline once startup completes
pipeline = None


def download_checkpoint_from_s3():
    """
    Downloads all objects under S3_BUCKET/S3_PREFIX into /tmp/checkpoint-final/,
    preserving folder structure. Skips any “folder placeholder” keys that end in “/”.
    """
    if not S3_BUCKET or not S3_PREFIX:
        raise RuntimeError("S3_BUCKET and S3_PREFIX must both be set in the environment.")

    # Build a boto3 S3 client using credentials from ENV
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    local_root = "/tmp/checkpoint-final"
    # If leftover from a previous run, remove it
    if os.path.exists(local_root):
        shutil.rmtree(local_root)
    os.makedirs(local_root, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    # Iterate over every page of results under that prefix
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. "roomify-checkpoint-final/unet/config.json"
            # Skip keys that end with "/" (S3 folder placeholders)
            if key.endswith("/"):
                continue

            # Compute the relative path under our prefix:
            # If S3_PREFIX = "roomify-checkpoint-final/"
            # and key = "roomify-checkpoint-final/unet/config.json",
            # then rel_path = "unet/config.json"
            rel_path = os.path.relpath(key, S3_PREFIX)

            # Build the full local path where we will store this file
            local_file_path = os.path.join(local_root, rel_path)

            # Ensure the directory exists
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download the object into that exact filename
            s3.download_file(S3_BUCKET, key, local_file_path)
            print(f"Downloaded s3://{S3_BUCKET}/{key} → {local_file_path}")

    print(f"✅ Finished downloading checkpoint into {local_root}")


def load_img2img_pipeline():
    """
    1. Downloads the fine-tuned components from S3 into /tmp/checkpoint-final/
    2. Instantiates a base StableDiffusionImg2ImgPipeline from Hugging Face Hub (v1.5),
       then overwrites its text_encoder, tokenizer, and unet with your fine-tuned files.
    3. Replaces its scheduler with UniPCMultistepScheduler.
    4. Moves the pipeline to the appropriate device (GPU if available).
    """
    global pipeline

    # 1) Download everything from S3 → /tmp/checkpoint-final/
    download_checkpoint_from_s3()
    local_root = "/tmp/checkpoint-final"

    # 2) Instantiate a base Img2Img pipeline (v1.5) from the Hub
    #    We will keep its original VAE, but overwrite text_encoder/tokenizer/unet below.
    base_model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )

    # 3) Overwrite the tokenizer & text_encoder with the fine-tuned versions:
    #    Our S3 structure contains:
    #      /tmp/checkpoint-final/text_encoder/config.json
    #      /tmp/checkpoint-final/text_encoder/pytorch_model.bin
    #      /tmp/checkpoint-final/tokenizer/...
    #      /tmp/checkpoint-final/unet/...
    pipe.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(local_root, "tokenizer"))
    pipe.text_encoder = CLIPTextModel.from_pretrained(os.path.join(local_root, "text_encoder"))

    # 4) Overwrite the UNet with our fine-tuned UNet:
    pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(local_root, "unet"))

    # 5) (Optional) If you wish to keep the VAE from the fine-tuned checkpoint, you could load it similarly:
    #    pipe.vae = AutoencoderKL.from_pretrained(os.path.join(local_root, "vae"))
    #    BUT often DreamBooth fine-tunes only U-Net + Text Encoder, leaving VAE untouched.
    #    In that case, the base model’s VAE is fine. We’ll leave pipe.vae as is.

    # 6) Swap in UniPCMultistepScheduler for faster sampling:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 7) Move the entire pipeline to GPU (if available) or CPU otherwise:
    pipe = pipe.to(DEVICE)

    pipeline = pipe
    print("✅ Img2Img pipeline loaded and ready.")


@app.on_event("startup")
def on_startup():
    """
    Runs at service startup:
      • Downloads checkpoint from S3
      • Loads/upgrades the pipeline
    """
    try:
        load_img2img_pipeline()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Img2Img pipeline: {e}")


@app.get("/health")
def health_check():
    """
    Simple health endpoint to verify the pipeline is loaded.
    """
    if pipeline is None:
        return {"status": "error", "detail": "Pipeline not loaded."}

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_dir": "/tmp/checkpoint-final",
        "fixed_strength": FIXED_STRENGTH,
        "fixed_guidance": FIXED_GUIDANCE,
        "fixed_steps": FIXED_STEPS,
    }


@app.post("/generate")
async def generate_image(prompt: str, image: UploadFile = File(...)):
    """
    Expects a multipart/form-data POST with:
      • prompt: a string (must include "<interiorx>")
      • image: the input room photo (JPEG or PNG)

    Returns: the generated image as a raw PNG in the response body.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Read the uploaded file into a PIL Image
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
            num_inference_steps=FIXED_STEPS,
        )
        generated = results.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Convert to in‐memory PNG bytes
    buffer = io.BytesIO()
    generated.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
