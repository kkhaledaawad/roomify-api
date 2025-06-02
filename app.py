import os
import boto3
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import base64
from io import BytesIO

app = FastAPI()

# 1) Environment vars (set these in Render dashboard later)
S3_BUCKET = os.getenv("my-roomify-models")                # e.g. "my-roomify-models"
S3_PREFIX = os.getenv("roomify-checkpoint-final/")                # e.g. "roomify/checkpoint-final"
MODEL_DIR = "/tmp/checkpoint-final"               # where we’ll store the model locally

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNET_DIR        = os.path.join(MODEL_DIR, "unet")
TEXT_ENCODER_DIR= os.path.join(MODEL_DIR, "text_encoder")
TOKENIZER_DIR   = os.path.join(MODEL_DIR, "tokenizer")

# 2) Download function
def download_checkpoint_from_s3():
    if os.path.isdir(MODEL_DIR) and os.path.isdir(UNET_DIR) \
       and os.path.isdir(TEXT_ENCODER_DIR) and os.path.isdir(TOKENIZER_DIR):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]                                     # e.g. "roomify/checkpoint-final/unet/config.json"
            rel_path = key[len(S3_PREFIX)+1:]                    # e.g. "unet/config.json"
            local_path = os.path.join(MODEL_DIR, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(S3_BUCKET, key, local_path)

# 3) On startup: download + load pipeline
@app.on_event("startup")
def load_pipeline():
    download_checkpoint_from_s3()
    global img2img
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
    )
    img2img.scheduler = UniPCMultistepScheduler.from_config(img2img.scheduler.config)
    img2img.unet = img2img.unet.__class__.from_pretrained(
        UNET_DIR,
        torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
    )
    img2img.text_encoder = CLIPTextModel.from_pretrained(
        TEXT_ENCODER_DIR,
        torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
    )
    img2img.tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
    img2img.to(DEVICE)
    img2img.enable_attention_slicing()

# 4) Img2Img endpoint (fixed strength/guidance/steps, only prompt + image)
FIXED_STRENGTH = 0.75
FIXED_GUIDANCE = 7.5
FIXED_STEPS = 30

def pil_to_base64(img: Image.Image) -> str:
    buff = BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode()

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    if "<interiorx>" not in prompt:
        return JSONResponse(status_code=400, content={"error":"Prompt must include <interiorx>."})
    contents = await image.read()
    init_img = Image.open(BytesIO(contents)).convert("RGB").resize((512,512))
    with torch.autocast(DEVICE.type if DEVICE.type=="cuda" else "cpu"):
        outs = img2img(
            prompt=prompt,
            image=init_img,
            strength=FIXED_STRENGTH,
            num_inference_steps=FIXED_STEPS,
            guidance_scale=FIXED_GUIDANCE,
        )
        out_img = outs.images[0]
    return {"image_base64": pil_to_base64(out_img)}

@app.get("/health")
async def health():
    return {"status":"ok","device":str(DEVICE)}
