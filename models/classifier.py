import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ── SETTINGS ──────────────────────────────────────────
PROCESSED_PATH = "data/processed"
TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
MODEL_SAVE_PATH = "models/classifier_resnet50.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 4

LABEL_MAP = {
    "Everyday Casual": 0,
    "Other": 1,
    "Formal": 2,
    "Semi Formal": 3
}

# ── DATASET CLASS ─────────────────────────────────────
class GarmentDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((row["filename"], row["group"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, group = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP.get(group, 4)

        if self.transform:
            image = self.transform(image)

        return image, label

# ── TRANSFORMS ────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── LOAD DATA ─────────────────────────────────────────
train_dataset = GarmentDataset(TRAIN_CSV, PROCESSED_PATH, train_transform)
val_dataset = GarmentDataset(VAL_CSV, PROCESSED_PATH, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# ── MODEL ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# ── LOSS & OPTIMIZER ──────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── TRAINING LOOP ─────────────────────────────────────
print("\nStarting training...")
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset) * 100

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset) * 100

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  Model saved! Best Val Acc: {best_val_acc:.1f}%")

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.1f}%")