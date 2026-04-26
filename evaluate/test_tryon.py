import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── PATHS ─────────────────────────────────────────────
VAL_CSV = "data/val.csv"
IMG_DIR = "data/processed"
MODEL_PATH = "models/classifier_resnet50.pth"
NUM_CLASSES = 5

# ── LABEL MAP ─────────────────────────────────────────
LABEL_MAP = {
    "Bridal": 0,
    "Everyday Casual": 1,
    "Semi Formal": 2,
    "Wedding Guest": 3,
    "Other": 4
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── TRANSFORM ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── LOAD MODEL ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded!")

# ── LOAD VAL DATA ─────────────────────────────────────
val_data = []
with open(VAL_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_data.append((row["filename"], row["group"]))

print(f"Validation samples: {len(val_data)}")

# ── EVALUATE ──────────────────────────────────────────
all_preds = []
all_labels = []

for filename, group in val_data:
    img_path = os.path.join(IMG_DIR, filename)
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(1).item()
        all_preds.append(pred)
        all_labels.append(LABEL_MAP.get(group, 4))
    except:
        pass

# ── RESULTS ───────────────────────────────────────────
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    all_labels, all_preds,
    target_names=list(LABEL_MAP.keys())
))

# ── CONFUSION MATRIX ──────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=list(LABEL_MAP.keys()),
            yticklabels=list(LABEL_MAP.keys()),
            cmap="Blues")
plt.title("Confusion Matrix - Garment Classifier")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("evaluate/confusion_matrix.png")
print("\nConfusion matrix saved to evaluate/confusion_matrix.png")