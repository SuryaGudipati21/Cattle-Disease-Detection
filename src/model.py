import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="mobilenet_v2", num_classes=2, pretrained=True):
    """
    Returns a pretrained model with the classifier head replaced.
    Supports: mobilenet_v2, efficientnet_b0, resnet18
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from mobilenet_v2, efficientnet_b0, resnet18")

    return model


if __name__ == "__main__":
    model = get_model("mobilenet_v2")
    print(model)
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # Should be [1, 2]