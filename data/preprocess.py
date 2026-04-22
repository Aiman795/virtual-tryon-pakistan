import json
import os
from PIL import Image

# Paths
JSON_PATH = "data/raw/Threads of Fashion.coco-segmentation/train/_annotations.coco.json"
TRAIN_PATH = "data/raw/Threads of Fashion.coco-segmentation/train"
OUTPUT_PATH = "data/processed"

# Target size
TARGET_SIZE = (768, 1024)

# Load JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Get all image filenames
images = data["images"]
total = len(images)

print(f"Total images to process: {total}")
print("Starting preprocessing...")

success = 0
failed = 0

for i, img_info in enumerate(images):
    filename = img_info["file_name"]
    input_path = os.path.join(TRAIN_PATH, filename)
    output_path = os.path.join(OUTPUT_PATH, filename)

    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(TARGET_SIZE)
        img.save(output_path)
        success += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total} images...")

    except Exception as e:
        print(f"Failed: {filename} — {e}")
        failed += 1

print(f"\nDone!")
print(f"Successfully processed: {success}")
print(f"Failed: {failed}")
print(f"Saved to: {OUTPUT_PATH}")