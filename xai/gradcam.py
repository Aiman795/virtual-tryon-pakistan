import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── SETTINGS ──────────────────────────────────────────
LABEL_MAP = {
    0: "Everyday Casual",
    1: "Other",
    2: "Formal",
    3: "Semi Formal"
}

MODEL_PATH = "models/classifier_resnet50_v2.pth"

# ── LOAD MODEL ────────────────────────────────────────
device = torch.device("cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── GRAD-CAM CLASS ────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:])

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.numpy()
        cam = cv2.resize(cam, (224, 224))

        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx


def generate_gradcam(image_path, output_path="data/gradcam_result.jpg"):
    """
    Generate Grad-CAM heatmap for a garment image
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    input_tensor = transform(img).unsqueeze(0)
    input_tensor.requires_grad = True

    # Initialize Grad-CAM on last conv layer
    target_layer = model.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    cam, class_idx = gradcam.generate(input_tensor)
    label = LABEL_MAP.get(class_idx, "Unknown")

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)

    # Create side by side figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_array)
    axes[0].set_title("Original Garment", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPredicted: {label}", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Explanation — AI focused on highlighted regions",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Grad-CAM saved to {output_path}")
    print(f"Predicted class: {label}")

    return output_path, label


# ── TEST ───────────────────────────────────────────────
if __name__ == "__main__":
    result, label = generate_gradcam("data/garment_test.jpg")
    print(f"Done! Result: {result}, Label: {label}")