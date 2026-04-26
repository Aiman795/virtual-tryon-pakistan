import json
import os
import csv
import random

# Paths
JSON_PATH = "data/raw/Threads of Fashion.coco-segmentation/train/_annotations.coco.json"
OUTPUT_CSV = "data/labels.csv"
TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"

# Load JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Category groups
GROUPS = {
    "Bridal": ["Bridal Choli", "Bridal Dupatta", "Bridal Frock", "Bridal Jacket",
               "Bridal Kameez", "Bridal Kurta", "Bridal Lehenga", "Bridal Maxi"],
    "Everyday Casual": ["Everyday Casual Bell Bottom", "Everyday Casual Chooridaar",
                        "Everyday Casual Chooridar", "Everyday Casual Culottes",
                        "Everyday Casual Dupatta", "Everyday Casual Frock",
                        "Everyday Casual Gharara", "Everyday Casual Jacket",
                        "Everyday Casual Kurta", "Everyday Casual Lehenga",
                        "Everyday Casual Maxi", "Everyday Casual Palazzo Pants",
                        "Everyday Casual Peplum", "Everyday Casual Shalwar",
                        "Everyday Casual Straight Pant", "Everyday Casual Trouser",
                        "Everyday Casual Trousers"],
    "Semi Formal": ["Semi-Formal Anghrakha", "Semi-Formal Bell Bottom",
                    "Semi-Formal Casual Dupatta", "Semi-Formal Choli",
                    "Semi-Formal Chooridar", "Semi-Formal Culottes",
                    "Semi-Formal Dupatta", "Semi-Formal Frock",
                    "Semi-Formal Gharara", "Semi-Formal Jacket",
                    "Semi-Formal Kurta", "Semi-Formal Lehenga",
                    "Semi-Formal Maxi", "Semi-Formal Palazzo Pants",
                    "Semi-Formal Saree", "Semi-Formal Sari",
                    "Semi-Formal Shalwar", "Semi-Formal Sharara",
                    "Semi-Formal Straight Pant", "Semi-Formal Trousers"],
    "Wedding Guest": ["Wedding Guest Angrakha", "Wedding Guest Bell Bottom",
                      "Wedding Guest Choli", "Wedding Guest Chooridar",
                      "Wedding Guest Culottes", "Wedding Guest Dupatta",
                      "Wedding Guest Frock", "Wedding Guest Gharara",
                      "Wedding Guest Jacket", "Wedding Guest Kaftan",
                      "Wedding Guest Kurta", "Wedding Guest Kurti",
                      "Wedding Guest Lehenga", "Wedding Guest Maxi",
                      "Wedding Guest Maxi'", "Wedding Guest Palazzo Pants",
                      "Wedding Guest Peplum", "Wedding Guest Sari",
                      "Wedding Guest Shalwar", "Wedding Guest Sharara",
                      "Wedding Guest Straight Pant", "Wedding Guest Trousers"],
    "Other": ["clothing", "name", "object"]
}

# Build category name to group mapping
cat_to_group = {}
for group, cats in GROUPS.items():
    for cat in cats:
        cat_to_group[cat] = group

# Build category id to name mapping
cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

# Build image id to filename mapping
img_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Build image id to category mapping from annotations
img_id_to_cat = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    cat_name = cat_id_to_name[ann["category_id"]]
    if img_id not in img_id_to_cat:
        img_id_to_cat[img_id] = cat_name

# Write labels CSV
rows = []
for img in data["images"]:
    img_id = img["id"]
    filename = img["file_name"]
    cat_name = img_id_to_cat.get(img_id, "Other")
    group = cat_to_group.get(cat_name, "Other")
    rows.append([filename, cat_name, group])

# Save full labels
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "category", "group"])
    writer.writerows(rows)

print(f"Total labeled images: {len(rows)}")

# Count per group
from collections import Counter
group_counts = Counter(r[2] for r in rows)
print("\nImages per group:")
for g, c in group_counts.items():
    print(f"  {g}: {c}")

# Split 80/20
random.seed(42)
random.shuffle(rows)
split = int(0.8 * len(rows))
train_rows = rows[:split]
val_rows = rows[split:]

with open(TRAIN_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "category", "group"])
    writer.writerows(train_rows)

with open(VAL_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "category", "group"])
    writer.writerows(val_rows)

print(f"\nTrain set: {len(train_rows)} images")
print(f"Validation set: {len(val_rows)} images")
print("Done! CSV files saved.")

# ── MERGE WEDDING GUEST + BRIDAL INTO FORMAL ──────────
import pandas as pd

print("\nMerging Wedding Guest + Bridal into Formal...")

for csv_file in ["data/labels.csv", "data/train.csv", "data/val.csv"]:
    df = pd.read_csv(csv_file)
    df["group"] = df["group"].replace({
        "Wedding Guest": "Formal",
        "Bridal": "Formal"
    })
    df.to_csv(csv_file, index=False)
    print(f"Updated: {csv_file}")

# Check new counts
df = pd.read_csv("data/labels.csv")
print("\nNew group counts:")
print(df["group"].value_counts())