from gradio_client import Client, handle_file
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import shutil

# Load segmentation model
SEG_MODEL = YOLO("models/best.pt")
LABELS = {0: "Formal", 1: "Everyday_Casual", 2: "Semi_Formal", 3: "Other"}

def segment_garment(image_path):
    """Extract clean garment using segmentation"""
    results = SEG_MODEL(image_path, conf=0.25)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not results or results[0].masks is None:
        print("No segmentation — using original")
        return image_path

    result = results[0]
    if not result.boxes or len(result.boxes) == 0:
        return image_path

    best_idx = result.boxes.conf.argmax().item()
    best_box = result.boxes[best_idx]
    best_mask = result.masks[best_idx]

    class_id = int(best_box.cls.item())
    label = LABELS.get(class_id, "Unknown")
    conf = best_box.conf.item()
    print(f"Segmented: {label} ({conf:.1%})")

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cropped = img_rgb[y1:y2, x1:x2]

    mask_data = best_mask.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_data, (img_rgb.shape[1], img_rgb.shape[0]))
    mask_crop = mask_resized[y1:y2, x1:x2]

    result_img = np.ones_like(cropped) * 255
    mask_bool = mask_crop > 0.5
    result_img[mask_bool] = cropped[mask_bool]

    seg_path = "data/segmented_garment.jpg"
    Image.fromarray(result_img).save(seg_path)
    return seg_path


def tryon(person_image_path, garment_image_path,
          output_path="data/tryon_result.jpg"):
    print("Step 1 — Segmenting garment...")
    clean_garment = segment_garment(garment_image_path)
    print(f"Using garment: {clean_garment}")

    print("Step 2 — Sending to IDM-VTON...")
    client = Client("yisol/IDM-VTON")

    result = client.predict(
        dict={"background": handle_file(person_image_path),
              "layers": [], "composite": None},
        garm_img=handle_file(clean_garment),
        garment_des="Pakistani garment",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    if result and result[0]:
        shutil.copy(result[0], output_path)
        print(f"Saved to {output_path}")
        return output_path
    return None


if __name__ == "__main__":
    PERSON_IMAGE = "data/person_test.jpg"
    GARMENT_IMAGE = "data/garment_test.jpg"
    result = tryon(PERSON_IMAGE, GARMENT_IMAGE)
    if result:
        print("Success!")