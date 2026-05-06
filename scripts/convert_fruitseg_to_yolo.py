import cv2
import shutil
import random
from pathlib import Path

RAW_DIR = Path(r"D:\codes\qa_cv\datasets\FruitSeg30")
OUT_DIR = Path(r"D:\codes\qa_cv\datasets\fruit_seg_yolo")

CLASS_ID = 0
CLASS_NAME = "fruit"

IMG_EXTS = [".jpg", ".jpeg", ".png"]


def find_masks_and_images():
    pairs = []

    for fruit_dir in RAW_DIR.iterdir():
        if not fruit_dir.is_dir():
            continue

        # Common names in this dataset
        possible_img_dirs = [
            fruit_dir / "images",
            fruit_dir / "image",
            fruit_dir / "Images",
            fruit_dir / "JPEGImages",
        ]

        possible_mask_dirs = [
            fruit_dir / "masks",
            fruit_dir / "mask",
            fruit_dir / "Masks",
            fruit_dir / "SegmentationClass",
        ]

        image_dir = next((p for p in possible_img_dirs if p.exists()), None)
        mask_dir = next((p for p in possible_mask_dirs if p.exists()), None)

        # fallback: if images/masks are directly inside fruit folder
        if image_dir is None:
            image_dir = fruit_dir
        if mask_dir is None:
            mask_dir = fruit_dir

        images = []
        masks = []

        for ext in IMG_EXTS:
            images.extend(image_dir.glob(f"*{ext}"))
            masks.extend(mask_dir.glob(f"*{ext}"))

        for mask_path in masks:
            mask_stem = mask_path.stem

            # remove common mask suffixes
            clean_stem = (
                mask_stem
                .replace("_mask", "")
                .replace("-mask", "")
                .replace(" mask", "")
                .replace("_segmentation", "")
            )

            img_path = None
            for ext in IMG_EXTS:
                candidates = [
                    image_dir / f"{clean_stem}{ext}",
                    image_dir / f"{mask_stem}{ext}",
                ]
                for c in candidates:
                    if c.exists() and c != mask_path:
                        img_path = c
                        break
                if img_path:
                    break

            if img_path:
                pairs.append((img_path, mask_path))

    return pairs


def mask_to_polygons(mask):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = binary.shape[:2]
    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        normalized = []
        for x, y in points:
            normalized.append(x / w)
            normalized.append(y / h)

        polygons.append(normalized)

    return polygons


def write_label(label_path, polygons):
    with open(label_path, "w") as f:
        for poly in polygons:
            values = [str(CLASS_ID)] + [f"{v:.6f}" for v in poly]
            f.write(" ".join(values) + "\n")


def main():
    pairs = find_masks_and_images()

    print(f"Found pairs: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split, items in splits.items():
        image_out_dir = OUT_DIR / split / "images"
        label_out_dir = OUT_DIR / split / "labels"

        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in items:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

            if img is None or mask is None:
                continue

            polygons = mask_to_polygons(mask)

            if not polygons:
                continue

            unique_name = f"{img_path.parent.parent.name}_{img_path.name}"

            out_img = image_out_dir / unique_name
            out_label = label_out_dir / f"{Path(unique_name).stem}.txt"

            shutil.copy2(img_path, out_img)
            write_label(out_label, polygons)

    data_yaml = f"""
path: {OUT_DIR.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: {CLASS_NAME}
"""

    (OUT_DIR / "data.yaml").write_text(data_yaml.strip())

    print(f"Done. YOLO dataset saved to: {OUT_DIR}")
    print(f"Train YAML: {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()