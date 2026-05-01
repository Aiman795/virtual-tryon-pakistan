import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# ── LOAD MODEL ────────────────────────────────────────
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

LABELS = {0: "Formal", 1: "Everyday_Casual", 2: "Semi_Formal", 3: "Other"}

def segment_garment(image_path, output_path="data/segmented_result.jpg"):
    """
    Run segmentation on a garment image
    Returns cropped garment image
    """
    print(f"Running segmentation on: {image_path}")

    # Run inference
    results = model(image_path, conf=0.25)

    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(results) == 0 or results[0].masks is None:
        print("No segmentation found — using original image")
        return image_path

    # Get best detection
    result = results[0]
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        print("No boxes found — using original image")
        return image_path

    # Get highest confidence detection
    best_idx = boxes.conf.argmax().item()
    best_box = boxes[best_idx]
    best_mask = masks[best_idx]

    # Get class and confidence
    class_id = int(best_box.cls.item())
    confidence = best_box.conf.item()
    label = LABELS.get(class_id, "Unknown")

    print(f"Detected: {label} ({confidence:.1%} confidence)")

    # Get bounding box
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    # Crop garment region
    cropped = img_rgb[y1:y2, x1:x2]

    # Apply mask to cropped region
    mask_data = best_mask.data[0].cpu().numpy()
    mask_resized = cv2.resize(
        mask_data,
        (img_rgb.shape[1], img_rgb.shape[0])
    )
    mask_crop = mask_resized[y1:y2, x1:x2]

    # Create white background
    result_img = np.ones_like(cropped) * 255
    mask_bool = mask_crop > 0.5
    result_img[mask_bool] = cropped[mask_bool]

    # Save result
    Image.fromarray(result_img).save(output_path)
    print(f"Segmented garment saved to: {output_path}")

    # Also save annotated image
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    Image.fromarray(annotated_rgb).save("data/annotated_result.jpg")
    print(f"Annotated image saved to: data/annotated_result.jpg")

    return output_path


# ── TEST ───────────────────────────────────────────────
if __name__ == "__main__":
    TEST_IMAGE = "data/garment_test.jpg"
    result = segment_garment(TEST_IMAGE)
    print(f"Done! Result: {result}")