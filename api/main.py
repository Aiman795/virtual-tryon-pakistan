from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
import uvicorn
import shutil
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from gradio_client import Client, handle_file
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xai.gradcam import generate_gradcam

sys.path.append(".")
from models.fit_score import recommend_size, generate_explanation

# ── APP SETUP ─────────────────────────────────────────
app = FastAPI(title="Virtual Try-On API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODELS ───────────────────────────────────────
print("Loading models...")

LABEL_MAP = {
    0: "Everyday Casual",
    1: "Other",
    2: "Formal",
    3: "Semi Formal"
}

classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 4)
classifier.load_state_dict(
    torch.load("models/classifier_resnet50_v2.pth", map_location="cpu")
)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

seg_model = YOLO("models/best.pt")

# ── GRADIO CLIENT WITH TIMEOUT ────────────────────────
# Increased timeout to handle slow connections / sleeping HF Spaces
try:
    tryon_client = Client(
        "yisol/IDM-VTON",
        httpx_kwargs={"timeout": 120}  # 120 seconds timeout
    )
    print("All models loaded!")
except Exception as e:
    print(f"Warning: Could not connect to IDM-VTON space: {e}")
    print("Try-on endpoint will attempt reconnection at request time.")
    tryon_client = None


# ── HELPER FUNCTIONS ──────────────────────────────────
def classify_garment(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = classifier(tensor)
        pred = output.argmax(1).item()
    return LABEL_MAP[pred]


def get_tryon_client():
    """Returns existing client or creates a new one if not initialized."""
    global tryon_client
    if tryon_client is None:
        print("Attempting to reconnect to IDM-VTON space...")
        tryon_client = Client(
            "yisol/IDM-VTON",
            httpx_kwargs={"timeout": 120}
        )
    return tryon_client


# ── ENDPOINTS ─────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Virtual Try-On API is running!"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    path = f"data/temp_{file.filename}"
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        category = classify_garment(path)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/fitscore")
async def fitscore(
    chest: float = Form(...),
    waist: float = Form(...),
    length: float = Form(...),
    brand: str = Form(...)
):
    try:
        size, score, breakdown, all_scores = recommend_size(
            chest, waist, length, brand
        )
        explanation = generate_explanation(chest, waist, length, brand)
        return {
            "recommended_size": size,
            "confidence": score,
            "all_scores": all_scores,
            "breakdown": breakdown,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fit score failed: {str(e)}")


@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    chest: float = Form(0),
    waist: float = Form(0),
    length: float = Form(0),
    brand: str = Form("Khaadi")
):
    person_path = "data/temp_person.jpg"
    garment_path = "data/temp_garment.jpg"

    # Save uploaded files
    try:
        with open(person_path, "wb") as f:
            shutil.copyfileobj(person.file, f)
        with open(garment_path, "wb") as f:
            shutil.copyfileobj(garment.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {str(e)}")

    # Step 1 — Classify garment
    try:
        category = classify_garment(garment_path)
    except Exception as e:
        category = "Everyday Casual"  # fallback
        print(f"Classification warning: {e}")

    # Step 2 — Try-On
    def call_tryon():
        client = get_tryon_client()
        return client.predict(
            dict={"background": handle_file(person_path),
                  "layers": [], "composite": None},
            garm_img=handle_file(garment_path),
            garment_des=f"Pakistani {category} garment",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

    tryon_error = None
    result = None
    try:
        result = await run_in_threadpool(call_tryon)
    except Exception as e:
        tryon_error = str(e)
        print(f"Try-on error: {e}")

    # Save output image if successful
    output_path = "data/tryon_output.jpg"
    tryon_image_url = None

    if result and result[0]:
        try:
            shutil.copy(result[0], output_path)
            tryon_image_url = "/result"
        except Exception as e:
            print(f"Failed to save output image: {e}")

    # Step 3 — Fit Score
    fit_result = {}
    if chest > 0 and waist > 0 and length > 0:
        try:
            size, score, breakdown, all_scores = recommend_size(
                chest, waist, length, brand
            )
            explanation = generate_explanation(chest, waist, length, brand)
            fit_result = {
                "recommended_size": size,
                "confidence": score,
                "explanation": explanation
            }
        except Exception as e:
            print(f"Fit score warning: {e}")

    # Return response (with error info if tryon failed)
    response = {
        "category": category,
        "tryon_image": tryon_image_url,
        "fit_score": fit_result
    }

    if tryon_error:
        response["tryon_error"] = f"Try-on failed (timeout or connection issue): {tryon_error}"

    return response


@app.get("/result")
def get_result():
    output_path = "data/tryon_output.jpg"
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="No result image found. Run /tryon first.")
    return FileResponse(output_path)

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    path = "data/temp_gradcam.jpg"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    def run_gradcam():
        return generate_gradcam(path, "data/gradcam_result.jpg")

    result_path, label = await run_in_threadpool(run_gradcam)

    return {
        "label": label,
        "gradcam_image": "/gradcam_result"
    }


@app.get("/gradcam_result")
def get_gradcam():
    return FileResponse("data/gradcam_result.jpg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)