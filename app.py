import os
import io
import base64
import boto3
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from starlette.middleware.cors import CORSMiddleware

# ----------------------------------------------------------------------
# READ ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")  # e.g., "roomify-checkpoint-final/"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

if not all([S3_BUCKET, S3_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
    raise RuntimeError(
        "One or more required environment variables are missing: "
        "S3_BUCKET, S3_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
    )

# ----------------------------------------------------------------------
# DEFINE GLOBAL VARIABLES
# ----------------------------------------------------------------------

TMP_CHECKPOINT_DIR = "/tmp/checkpoint-final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE: StableDiffusionImg2ImgPipeline = None  # Will be initialized on startup

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
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip if it's the prefix folder itself
            if key.endswith("/"):
                continue

            # Compute relative path inside TMP_CHECKPOINT_DIR
            # E.g. if S3_PREFIX="roomify-checkpoint-final/" and key="roomify-checkpoint-final/unet/config.json",
            # then relative_path="unet/config.json"
            relative_path = key[len(S3_PREFIX):]
            local_path = os.path.join(TMP_CHECKPOINT_DIR, relative_path)

            # Make sure local directory exists
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download
            s3.download_file(S3_BUCKET, key, local_path)
            downloaded_any = True

    if not downloaded_any:
        raise RuntimeError(f"No files found under s3://{S3_BUCKET}/{S3_PREFIX}")

# ----------------------------------------------------------------------
# LIFESPAN EVENT: on_startup → download + load pipeline
# ----------------------------------------------------------------------

@app.on_event("startup")
def load_img2img_pipeline():
    global PIPELINE

    # 1) Download the fine-tuned checkpoint into TMP_CHECKPOINT_DIR
    try:
        download_checkpoint_from_s3()
        print(f"✅ Finished downloading checkpoint into {TMP_CHECKPOINT_DIR}")
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint from S3: {e}")

    # 2) Load the StableDiffusionImg2ImgPipeline from TMP_CHECKPOINT_DIR
    try:
        # If you trained with DreamBooth, your folder likely has:
        # ├─ text_encoder/
        # ├─ tokenizer/
        # ├─ unet/
        # └─ model_index.json      ← must exist for from_pretrained to work
        PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
            TMP_CHECKPOINT_DIR,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            safety_checker=None,
        )

        # Replace the default scheduler with UniPC Multistep (optional but recommended)
        PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config)

        # Move pipeline to device
        PIPELINE.to(DEVICE)

        # If using xFormers for memory efficiency (CUDA-only), uncomment:
        # if DEVICE.type == "cuda":
        #     PIPELINE.enable_xformers_memory_efficient_attention()

        print("🚀 Img2Img pipeline loaded and moved to device.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Img2Img pipeline: {e}")

# ----------------------------------------------------------------------
# ROUTE: /generate → perform img2img given an input image and prompt
# ----------------------------------------------------------------------

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
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="The pipeline is not ready yet. Try again in a moment.")

    # 1) Read and preprocess the uploaded image
    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Diffusers expects 512×512 for img2img
        input_image = input_image.resize((512, 512), Image.LANCZOS)
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

# ----------------------------------------------------------------------
# HEALTHCHECK or ROOT ENDPOINT
# ----------------------------------------------------------------------

@app.get("/", summary="Simple healthcheck")
def read_root():
    return {"message": "Roomify Img2Img API is up and running!"}
