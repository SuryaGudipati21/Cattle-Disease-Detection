import os
import sys
import csv
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict import load_model, predict

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


# Chart + CSV helpers

def save_summary_chart(healthy_count, sick_count, save_path):
    """Pie + bar chart summary of batch predictions."""
    labels = ["Healthy", "Possibly Sick"]
    counts = [healthy_count, sick_count]
    colors = ["#2e7d32", "#c62828"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction Summary", fontsize=15, fontweight="bold")

    axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12}
    )
    axes[0].set_title("Distribution")

    bars = axes[1].bar(labels, counts, color=colors, width=0.4, edgecolor="white")
    axes[1].set_title("Count")
    axes[1].set_ylabel("Number of Images")
    for bar, count in zip(bars, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count), ha="center", fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Summary chart saved → {save_path}")


def save_csv(results, save_path):
    """Save batch results to CSV."""
    with open(save_path, "w", newline="") as f:
        fieldnames = ["image", "prediction", "confidence", "healthy_prob", "sick_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image":        r["filename"],
                "prediction":   r["label"],
                "confidence":   f"{r['confidence']:.2%}",
                "healthy_prob": f"{r['probs']['Healthy']:.2%}",
                "sick_prob":    f"{r['probs']['Possibly Sick']:.2%}",
            })
    print(f"📄 CSV saved → {save_path}")


# Main callable — used by both app.py and CLI

def run_batch(image_paths, model, output_dir):
    """
    Run predictions on a list of absolute image paths.
    Saves summary chart + CSV into output_dir.

    Returns a dict:
        {
            "results":       [ {filename, orig_img(=abs path), label, confidence, probs}, ... ],
            "healthy_count": int,
            "sick_count":    int,
            "summary_path":  str (abs path to chart PNG),
            "csv_path":      str (abs path to CSV),
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    results       = []
    healthy_count = 0
    sick_count    = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            label, confidence, probs = predict(img_path, model)
            results.append({
                "filename":   filename,
                "orig_img":   img_path,
                "label":      label,
                "confidence": confidence,
                "probs":      probs,
            })
            if label == "Healthy":
                healthy_count += 1
            else:
                sick_count += 1
        except Exception as e:
            print(f"[SKIP] {filename} — {e}")

    # Save outputs
    summary_path = os.path.join(output_dir, f"predictions_summary_{uuid.uuid4().hex}.png")
    csv_path     = os.path.join(output_dir, f"batch_results_{uuid.uuid4().hex}.csv")

    save_summary_chart(healthy_count, sick_count, summary_path)
    save_csv(results, csv_path)

    total = healthy_count + sick_count
    print(f"\n{'='*45}")
    print(f"✅ Healthy       : {healthy_count} / {total}")
    print(f"⚠️  Possibly Sick : {sick_count} / {total}")
    print(f"{'='*45}")

    return {
        "results":       results,
        "healthy_count": healthy_count,
        "sick_count":    sick_count,
        "summary_path":  summary_path,
        "csv_path":      csv_path,
    }

# CLI entry point (unchanged behaviour)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else input("Enter folder path: ").strip()

    if not os.path.isdir(folder):
        print(f"❌ Not a valid folder: {folder}")
        sys.exit(1)

    image_paths = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    if not image_paths:
        print(f"❌ No images found in {folder}")
        sys.exit(1)

    print(f"📂 Found {len(image_paths)} images in {folder}\n")

    model = load_model()
    run_batch(image_paths, model, output_dir=folder)