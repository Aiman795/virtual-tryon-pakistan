from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Classifier
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

# Segmentation
seg_model = YOLO("models/best.pt")
SEG_LABELS = {0: "Formal", 1: "Everyday_Casual", 2: "Semi_Formal", 3: "Other"}

# Try-On Client
tryon_client = Client("yisol/IDM-VTON")

print("All models loaded!")

# ── HELPER FUNCTIONS ──────────────────────────────────
def classify_garment(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = classifier(tensor)
        pred = output.argmax(1).item()
    return LABEL_MAP[pred]


def segment_garment(image_path):
    results = seg_model(image_path, conf=0.25)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not results or results[0].masks is None:
        return image_path

    result = results[0]
    if not result.boxes or len(result.boxes) == 0:
        return image_path

    best_idx = result.boxes.conf.argmax().item()
    best_box = result.boxes[best_idx]
    best_mask = result.masks[best_idx]

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cropped = img_rgb[y1:y2, x1:x2]

    mask_data = best_mask.data[0].cpu().numpy()
    mask_resized = cv2.resize(
        mask_data, (img_rgb.shape[1], img_rgb.shape[0])
    )
    mask_crop = mask_resized[y1:y2, x1:x2]

    result_img = np.ones_like(cropped) * 255
    mask_bool = mask_crop > 0.5
    result_img[mask_bool] = cropped[mask_bool]

    seg_path = "data/temp_segmented.jpg"
    Image.fromarray(result_img).save(seg_path)
    return seg_path


# ── ENDPOINTS ─────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Virtual Try-On API is running!"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Save uploaded file
    path = f"data/temp_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Classify
    category = classify_garment(path)
    os.remove(path)

    return {"category": category}


@app.post("/fitscore")
async def fitscore(
    chest: float = Form(...),
    waist: float = Form(...),
    length: float = Form(...),
    brand: str = Form(...)
):
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


@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    chest: float = Form(0),
    waist: float = Form(0),
    length: float = Form(0),
    brand: str = Form("Khaadi")
):
    # Save uploads
    person_path = f"data/temp_person.jpg"
    garment_path = f"data/temp_garment.jpg"

    with open(person_path, "wb") as f:
        shutil.copyfileobj(person.file, f)
    with open(garment_path, "wb") as f:
        shutil.copyfileobj(garment.file, f)

    # Step 1 — Classify garment
    category = classify_garment(garment_path)

    # Step 2 — Segment garment
    clean_garment = segment_garment(garment_path)

    # Step 3 — Try-On
    result = tryon_client.predict(
        dict={"background": handle_file(person_path),
              "layers": [], "composite": None},
        garm_img=handle_file(clean_garment),
        garment_des=f"Pakistani {category} garment",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    output_path = "data/tryon_output.jpg"
    if result and result[0]:
        shutil.copy(result[0], output_path)

    # Step 4 — Fit Score
    fit_result = {}
    if chest > 0 and waist > 0 and length > 0:
        size, score, breakdown, all_scores = recommend_size(
            chest, waist, length, brand
        )
        explanation = generate_explanation(chest, waist, length, brand)
        fit_result = {
            "recommended_size": size,
            "confidence": score,
            "explanation": explanation
        }

    return {
        "category": category,
        "tryon_image": "/result",
        "fit_score": fit_result
    }


@app.get("/result")
def get_result():
    return FileResponse("data/tryon_output.jpg")


# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)