import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from model import get_model

CLASS_NAMES = ["Healthy", "Possibly Sick"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(model_path="models/best_model.pth", model_name="mobilenet_v2"):
    model = get_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model):
    """
    Returns: (label_str, confidence_float, probabilities_dict)
    """
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx    = int(np.argmax(probs))
    label       = CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])
    prob_dict   = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

    return label, confidence, prob_dict


def generate_gradcam(image_path, model, target_layer=None):
    """
    Generates a Grad-CAM heatmap overlay.
    Works with MobileNetV2 by default (last conv layer).
    Returns: overlaid image as numpy array (BGR)
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    img_pil  = Image.open(image_path).convert("RGB")
    img_np   = np.array(img_pil.resize((224, 224))) / 255.0
    tensor   = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    # Default: last conv layer of MobileNetV2
    if target_layer is None:
        target_layer = model.features[-1]

    cam    = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)[0]
    overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    return overlay


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    model = load_model()
    label, conf, probs = predict(img_path, model)

    print(f"\nImage     : {img_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2%}")
    print(f"Probabilities:")
    for cls, p in probs.items():
        print(f"   {cls}: {p:.2%}")