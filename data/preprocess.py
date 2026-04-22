import json
import os
from collections import Counter

# Path to your dataset
JSON_PATH = "data/raw/Threads of Fashion.coco-segmentation/train/_annotations.coco.json"
TRAIN_PATH = "data/raw/Threads of Fashion.coco-segmentation/train"

# Load the JSON file
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Print all categories
print("=== ALL CATEGORIES ===")
categories = data["categories"]
for cat in categories:
    print(f"ID: {cat['id']}  Name: {cat['name']}")

print(f"\nTotal categories: {len(categories)}")

# Count images per category
print("\n=== IMAGE COUNT PER CATEGORY ===")
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
annotation_counts = Counter()

for ann in data["annotations"]:
    annotation_counts[ann["category_id"]] += 1

for cat_id, count in sorted(annotation_counts.items()):
    print(f"{cat_id_to_name[cat_id]}: {count} images")

# Total images
print(f"\nTotal images in dataset: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")