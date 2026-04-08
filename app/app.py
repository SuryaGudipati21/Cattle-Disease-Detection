import os
import sys
import uuid

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image as PILImage

# ----------------------------Add src/ to path --------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_model, predict, generate_gradcam
from batch_predict_visual import run_batch

app = Flask(__name__, template_folder='ui')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL = None  # Lazy-loaded on first request


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(
            model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
        )
    return MODEL



# ROUTES

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/upload/single")
def upload_single():
    return render_template("single.html")


@app.route("/upload/folder")
def upload_folder():
    return render_template("folder.html")


@app.route("/predict/single", methods=["POST"])
def predict_single():
    """Calls predict() and generate_gradcam() from predict.py."""
    file = request.files.get("image")
    if not file:
        return redirect(url_for("upload_single"))

    fname     = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(save_path)

    model                    = get_model()
    label, confidence, probs = predict(save_path, model)

    cam_fname = None
    try:
        cam_array = generate_gradcam(save_path, model)
        cam_fname = f"cam_{fname}"
        PILImage.fromarray(cam_array).save(
            os.path.join(app.config['UPLOAD_FOLDER'], cam_fname)
        )
    except Exception as e:
        print(f"[Grad-CAM skipped] {e}")

    return render_template(
        "single.html",
        label=label, confidence=confidence, probs=probs,
        orig_img=fname, cam_img=cam_fname,
    )


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Saves uploaded files, then delegates everything to run_batch() from batch_predict_visual.py."""
    files = request.files.getlist("images")
    if not files:
        return redirect(url_for("upload_folder"))

    # Save uploaded files and keep mapping: abs_path -> saved_fname
    saved = []  # (original_filename, saved_fname, abs_path)
    for file in files:
        if not file.filename:
            continue
        saved_fname = f"{uuid.uuid4().hex}_{file.filename}"
        abs_path    = os.path.join(app.config['UPLOAD_FOLDER'], saved_fname)
        file.save(abs_path)
        saved.append((file.filename, saved_fname, abs_path))

    if not saved:
        return redirect(url_for("upload_folder"))

    # Delegate to batch_predict_visual.run_batch()
    image_paths = [abs_path for _, _, abs_path in saved]
    batch_out   = run_batch(image_paths, get_model(), output_dir=app.config['UPLOAD_FOLDER'])

    # run_batch stores abs path in orig_img -- map back to web-accessible saved_fname
    path_to_fname = {abs_path: saved_fname for _, saved_fname, abs_path in saved}
    for r in batch_out["results"]:
        r["saved_fname"] = path_to_fname.get(r["orig_img"], os.path.basename(r["orig_img"]))

    return render_template(
        "batch_results.html",
        results=batch_out["results"],
        total=len(batch_out["results"]),
        healthy_count=batch_out["healthy_count"],
        sick_count=batch_out["sick_count"],
        summary_img=os.path.basename(batch_out["summary_path"]),
        csv_file=os.path.basename(batch_out["csv_path"]),
    )


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)