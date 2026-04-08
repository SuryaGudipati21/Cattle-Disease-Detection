import os
import random
import shutil
import csv
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

# Config

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

# Blur detection
# How it works: Laplacian variance measures edge sharpness.
# LOWER threshold = LESS strict = KEEP more images.
# Start at 20.0 — only rejects severely blurry/out-of-focus images.
# Raise to 50 if you still want to cut more, lower to 10 if losing too many.
DEFAULT_BLUR_THRESHOLD = 20.0

# Whether to also downscale before blur check (helps for high-res images
# where Laplacian variance is artificially inflated).
BLUR_CHECK_RESIZE_TO = 640   # resize longest edge to this before checking; None = no resize

# Helpers

def _is_image(fname: str) -> bool:
    return fname.lower().endswith(SUPPORTED_EXTS)


def _collect_views(class_dir: str) -> dict:
    """
    Returns dict: view_name -> [list of absolute image paths]

    Flat structure  (healthy/*.jpg)         -> {"_all": [...]}
    View structure  (healthy/body/*.jpg)    -> {"body": [...], "nose": [...], ...}
    """
    views = {}
    entries = os.listdir(class_dir)
    has_subdirs = any(os.path.isdir(os.path.join(class_dir, e)) for e in entries)

    if has_subdirs:
        for entry in sorted(entries):
            sub = os.path.join(class_dir, entry)
            if not os.path.isdir(sub):
                continue
            imgs = sorted([os.path.join(sub, f) for f in os.listdir(sub) if _is_image(f)])
            if imgs:
                views[entry] = imgs
    else:
        imgs = sorted([os.path.join(class_dir, f) for f in entries if _is_image(f)])
        if imgs:
            views["_all"] = imgs

    return views


def _laplacian_variance(img_gray, resize_to=None):
    """Compute Laplacian variance of a grayscale image."""
    if resize_to is not None:
        h, w = img_gray.shape[:2]
        scale = resize_to / max(h, w)
        if scale < 1.0:
            img_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def is_blurry(image_path: str, threshold: float = DEFAULT_BLUR_THRESHOLD) -> tuple[bool, float]:
    """
    Returns (is_blurry: bool, score: float).
    score < threshold  →  blurry.
    Only flags images that are severely out-of-focus.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True, 0.0
    score = _laplacian_variance(img, resize_to=BLUR_CHECK_RESIZE_TO)
    return score < threshold, round(score, 2)


def _setup_log_dir(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# Step 1 — Clean

def clean_dataset(
    src_dir: str,
    dst_dir: str,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    log_dir: str = "logs",
):
    """
    Copies clean images from src_dir → dst_dir.
    Only removes:
      - Images that cv2 cannot open at all (truly corrupt/unreadable)
      - Images so blurry they score below `blur_threshold` (default=20)
    Preserves full subfolder structure.
    Writes logs to logs/blur_removed.txt and logs/corrupt_removed.txt.
    """
    _setup_log_dir(log_dir)

    blur_log    = open(os.path.join(log_dir, "blur_removed.txt"), "w")
    corrupt_log = open(os.path.join(log_dir, "corrupt_removed.txt"), "w")

    removed_blur    = 0
    removed_corrupt = 0
    kept            = 0

    print(f"\n{'─'*60}")
    print(f"  STEP 1 — Cleaning  (blur threshold = {blur_threshold})")
    print(f"  Source : {src_dir}")
    print(f"  Output : {dst_dir}")
    print(f"{'─'*60}\n")

    for class_name in sorted(os.listdir(src_dir)):
        src_class = os.path.join(src_dir, class_name)
        if not os.path.isdir(src_class):
            continue

        views = _collect_views(src_class)

        for view_name, paths in views.items():
            if view_name == "_all":
                dst_folder = os.path.join(dst_dir, class_name)
            else:
                dst_folder = os.path.join(dst_dir, class_name, view_name)
            os.makedirs(dst_folder, exist_ok=True)

            tag = f"{class_name}/{view_name}" if view_name != "_all" else class_name
            view_kept = view_blur = view_corrupt = 0

            for src_path in tqdm(paths, desc=f"  Cleaning [{tag}]"):
                fname    = os.path.basename(src_path)
                dst_path = os.path.join(dst_folder, fname)

                # ── Corruption check (open, not verify — verify consumes the handle) ──
                try:
                    with Image.open(src_path) as im:
                        im.load()          # forces full decode
                except Exception as e:
                    corrupt_log.write(f"{src_path}\t{e}\n")
                    removed_corrupt += 1
                    view_corrupt += 1
                    continue

                # ── Blur check ──
                blurry, score = is_blurry(src_path, blur_threshold)
                if blurry:
                    blur_log.write(f"{src_path}\tscore={score}\n")
                    removed_blur += 1
                    view_blur += 1
                    continue

                shutil.copy2(src_path, dst_path)
                kept += 1
                view_kept += 1

            print(f"  [{tag}] kept={view_kept} | blurry={view_blur} | corrupt={view_corrupt}  (total in view: {len(paths)})")

    blur_log.close()
    corrupt_log.close()

    print(f"\n{'─'*60}")
    print(f"  ✅  Clean done")
    print(f"      Kept            : {kept}")
    print(f"      Removed (blurry): {removed_blur}  → see {log_dir}/blur_removed.txt")
    print(f"      Removed (corrupt): {removed_corrupt}  → see {log_dir}/corrupt_removed.txt")
    print(f"{'─'*60}\n")


# Step 2 — Validate views

def _validate_views(processed_dir: str) -> set | None:
    """
    Checks every view exists in ALL classes.
    Views missing from any class are excluded so the model
    cannot use view type as a class shortcut.
    Returns safe common views, or None for flat structures.
    """
    class_views = {}
    for class_name in sorted(os.listdir(processed_dir)):
        class_path = os.path.join(processed_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        views = _collect_views(class_path)
        class_views[class_name] = set(views.keys())

    if not class_views:
        return None

    all_views    = set.union(*class_views.values())
    common_views = set.intersection(*class_views.values())
    missing      = all_views - common_views

    print("---- View validation -----------------------------")
    for cls, views in class_views.items():
        print(f"  {cls}: {sorted(views)}")

    if missing:
        print(f"\n  ⚠️  Views missing from at least one class — will be SKIPPED:")
        for view in sorted(missing):
            owners = [c for c, v in class_views.items() if view in v]
            print(f"     '{view}' only in: {owners}")
    else:
        print(f"\n  ✅ All views present in every class: {sorted(common_views)}")

    print("-----------------------------------------\n")

    if common_views == {"_all"}:
        return None
    return common_views


# Step 3 — Split

def train_val_split(
    processed_dir: str,
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    log_dir: str = "logs",
    balance: bool = False,          # OFF by default — set True only if you explicitly want it
    balance_per_view: bool = False, # if balance=True, balance across views within a class
):
    _setup_log_dir(log_dir)
    random.seed(seed)

    safe_views = _validate_views(processed_dir)

    # Collect everything first so we can do class-level balancing if needed
    class_images = {}   # class_name -> list of (src_path, dest_fname)

    for class_name in sorted(os.listdir(processed_dir)):
        src_class = os.path.join(processed_dir, class_name)
        if not os.path.isdir(src_class):
            continue

        views = _collect_views(src_class)

        if safe_views is not None:
            views = {v: p for v, p in views.items() if v in safe_views}

        if not views:
            print(f"[{class_name}] No usable views found — skipping.")
            continue

        all_entries = []   # (src_path, dest_fname)

        if balance and balance_per_view and len(views) > 1:
            # Old behaviour — cap per view (highest data loss, not recommended)
            min_count = min(len(p) for p in views.values())
            print(f"[{class_name}] Per-view balancing → capping each view to {min_count}")
            for view_name, paths in views.items():
                random.shuffle(paths)
                for src in paths[:min_count]:
                    prefix = f"{view_name}_" if view_name != "_all" else ""
                    all_entries.append((src, prefix + os.path.basename(src)))
        else:
            for view_name, paths in views.items():
                prefix = f"{view_name}_" if view_name != "_all" else ""
                for src in paths:
                    all_entries.append((src, prefix + os.path.basename(src)))

        class_images[class_name] = all_entries

    # ---- Optional class-level balancing -----------------------------------------
    if balance and not balance_per_view:
        min_class = min(len(v) for v in class_images.values())
        print(f"\n⚖️  Class balancing: capping all classes to {min_class} images each")
        for cls in class_images:
            entries = class_images[cls]
            random.shuffle(entries)
            class_images[cls] = entries[:min_class]

    # --- Split and copy -----------------------------------------
    summary_rows = []

    for class_name, entries in class_images.items():
        random.shuffle(entries)
        split_idx   = max(1, int(len(entries) * (1 - val_ratio)))
        train_items = entries[:split_idx]
        val_items   = entries[split_idx:]

        dst_train = os.path.join(train_dir, class_name)
        dst_val   = os.path.join(val_dir,   class_name)
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val,   exist_ok=True)

        for src, fname in tqdm(train_items, desc=f"  Copy train [{class_name}]"):
            shutil.copy2(src, os.path.join(dst_train, fname))

        for src, fname in tqdm(val_items, desc=f"  Copy val   [{class_name}]"):
            shutil.copy2(src, os.path.join(dst_val, fname))

        print(f"  [{class_name}]  total={len(entries)} → train={len(train_items)} | val={len(val_items)}")
        summary_rows.append((class_name, len(entries), len(train_items), len(val_items)))

    # --- Write summary log ---------------------------------------
    summary_path = os.path.join(log_dir, "split_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "total_clean", "train", "val"])
        w.writerows(summary_rows)

    print(f"\nSplit summary saved → {summary_path}")
    print("Split done.")
    _print_final_summary(train_dir, val_dir)


def _print_final_summary(train_dir: str, val_dir: str):
    print("\n---- Final dataset summary -----------------------------")
    grand_train = grand_val = 0
    for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
        if not os.path.exists(split_dir):
            continue
        print(f"  {split_name}/")
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            count = sum(1 for f in os.listdir(class_path) if _is_image(f))
            print(f"    {class_name}/  →  {count} images")
            if split_name == "train":
                grand_train += count
            else:
                grand_val += count
    print(f"\n  GRAND TOTAL  →  train: {grand_train} | val: {grand_val} | combined: {grand_train + grand_val}")
    print("-----------------------------------------------------\n")


# --- CLI entry point ----------------------------------------------------------

if __name__ == "__main__":

    # Paths
    RAW_DIR       = "data/raw"
    PROCESSED_DIR = "data/processed"
    TRAIN_DIR     = "data/train"
    VAL_DIR       = "data/val"
    LOG_DIR       = "logs"

    # Tuning
    # BLUR_THRESHOLD:
    #   20  → removes only severely blurry images (recommended — safe default)
    #   50  → moderate; removes noticeably soft images
    #   100 → aggressive (old value — removes far too many)
    #   0   → disables blur filtering entirely
    BLUR_THRESHOLD = 20.0

    # VAL_RATIO: fraction of images held out for validation
    VAL_RATIO = 0.2

    # BALANCE:
    #   False (recommended) → use all clean images, handle imbalance via loss weights
    #   True  → caps classes/views to smallest count (causes data loss)
    BALANCE = False

    # Run 
    clean_dataset(
        src_dir=RAW_DIR,
        dst_dir=PROCESSED_DIR,
        blur_threshold=BLUR_THRESHOLD,
        log_dir=LOG_DIR,
    )

    train_val_split(
        processed_dir=PROCESSED_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        val_ratio=VAL_RATIO,
        seed=42,
        log_dir=LOG_DIR,
        balance=BALANCE,
        balance_per_view=False,
    )

    print("\nAll done! Your dataset is ready in data/train/ and data/val/")
    print(f"Logs written to {LOG_DIR}/")
    print(f"\nIf you still lose more images than expected:")
    print(f"1. Open logs/blur_removed.txt and spot-check a few images visually.")
    print(f"2. Lower BLUR_THRESHOLD further (try 10 or even 5).")
    print(f"3. Set BLUR_THRESHOLD=0 to skip blur filtering entirely.")