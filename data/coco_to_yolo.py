import json
import os
import shutil
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────
JSON_PATH = "data/raw/Threads of Fashion.coco-segmentation/train/_annotations.coco.json"
IMAGES_PATH = "data/raw/Threads of Fashion.coco-segmentation/train"
OUTPUT_PATH = "data/yolo_dataset"

# ── CREATE FOLDERS ────────────────────────────────────
os.makedirs(f"{OUTPUT_PATH}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/val", exist_ok=True)

# ── LOAD JSON ─────────────────────────────────────────
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# ── BUILD MAPPINGS ────────────────────────────────────
categories = {cat["id"]: cat["name"] for cat in data["categories"]}
images = {img["id"]: img for img in data["images"]}

# ── GROUP ANNOTATIONS BY IMAGE ────────────────────────
from collections import defaultdict
img_annotations = defaultdict(list)
for ann in data["annotations"]:
    img_annotations[ann["image_id"]].append(ann)

# ── CATEGORY TO GROUP MAPPING ─────────────────────────
def get_group_id(cat_name):
    if "Bridal" in cat_name or "Wedding" in cat_name:
        return 0  # Formal
    elif "Everyday Casual" in cat_name:
        return 1  # Everyday Casual
    elif "Semi-Formal" in cat_name:
        return 2  # Semi Formal
    else:
        return 3  # Other

# ── CONVERT TO YOLO FORMAT ────────────────────────────
print("Converting annotations...")

image_ids = list(images.keys())
split = int(0.8 * len(image_ids))
train_ids = set(image_ids[:split])
val_ids = set(image_ids[split:])

train_count = 0
val_count = 0
skip_count = 0

for img_id, img_info in images.items():
    filename = img_info["file_name"]
    img_w = img_info["width"]
    img_h = img_info["height"]

    anns = img_annotations[img_id]
    if not anns:
        skip_count += 1
        continue

    # Determine split
    split_name = "train" if img_id in train_ids else "val"

    # Copy image
    src = os.path.join(IMAGES_PATH, filename)
    dst = f"{OUTPUT_PATH}/images/{split_name}/{filename}"
    if os.path.exists(src):
        shutil.copy(src, dst)

    # Write label file
    label_file = f"{OUTPUT_PATH}/labels/{split_name}/{Path(filename).stem}.txt"
    with open(label_file, "w") as f:
        for ann in anns:
            cat_name = categories[ann["category_id"]]
            group_id = get_group_id(cat_name)

            # Convert segmentation to YOLO format
            if ann.get("segmentation") and len(ann["segmentation"]) > 0:
                seg = ann["segmentation"][0]
                # Normalize coordinates
                normalized = []
                for i in range(0, len(seg), 2):
                    x = seg[i] / img_w
                    y = seg[i+1] / img_h
                    normalized.extend([x, y])

                if len(normalized) >= 6:
                    coords = " ".join([f"{v:.6f}" for v in normalized])
                    f.write(f"{group_id} {coords}\n")

    if split_name == "train":
        train_count += 1
    else:
        val_count += 1

print(f"Train images: {train_count}")
print(f"Val images: {val_count}")
print(f"Skipped: {skip_count}")

# ── CREATE YAML FILE ──────────────────────────────────
yaml_content = f"""path: /kaggle/input/datasets/aimanabbasi123/pakistani-yolo-dataset/yolo_dataset
train: images/train
val: images/val

nc: 4
names:
  0: Formal
  1: Everyday_Casual
  2: Semi_Formal
  3: Other
"""

with open(f"{OUTPUT_PATH}/dataset.yaml", "w") as f:
    f.write(yaml_content)

print("YAML file created!")
print(f"Dataset saved to {OUTPUT_PATH}")
print("Done!")