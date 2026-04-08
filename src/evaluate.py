import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model

CLASS_NAMES = ["Healthy", "Possibly Sick"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictions(model, loader):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved → {save_path}")


def print_metrics(y_true, y_pred):
    print("\nEvaluation Metrics")
    print("=" * 40)
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def main():
    _, val_loader = get_dataloaders("data", batch_size=32)

    model = get_model("mobilenet_v2", num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    y_true, y_pred = get_predictions(model, val_loader)

    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)

    print("\nAll evaluation files saved to models/")


if __name__ == "__main__":
    main()