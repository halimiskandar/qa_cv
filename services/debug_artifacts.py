import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _save_rgb(path: str, img_rgb: np.ndarray) -> None:
    cv2.imwrite(path, _rgb_to_bgr(img_rgb))


def _save_mask(path: str, mask: np.ndarray) -> None:
    if mask is None:
        return
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    cv2.imwrite(path, mask)


def _resize_mask(mask: np.ndarray, shape_hw) -> np.ndarray:
    if mask is None:
        return None
    h, w = shape_hw
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def create_mask_overlay(
    crop_rgb: np.ndarray,
    masks: dict,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Creates a visual overlay for demos/debugging.

    Colors are intentionally explicit because this is a debug image:
    yellow mask = yellow, green mask = green, brown/black defects = red,
    sticker/logo mask = blue.
    """
    overlay = crop_rgb.copy().astype(np.float32)
    h, w = crop_rgb.shape[:2]

    color_map = {
        "yellow_mask": np.array([255, 255, 0], dtype=np.float32),
        "green_mask": np.array([0, 255, 0], dtype=np.float32),
        "brown_black_mask": np.array([255, 0, 0], dtype=np.float32),
        "sticker_mask": np.array([0, 120, 255], dtype=np.float32),
        "dark_cluster_mask": np.array([255, 0, 255], dtype=np.float32),
    }

    # Draw lower-priority masks first, defect masks last.
    order = ["yellow_mask", "green_mask", "sticker_mask", "brown_black_mask", "dark_cluster_mask"]

    for name in order:
        mask = _resize_mask(masks.get(name), (h, w))
        if mask is None:
            continue
        active = mask > 0
        if not np.any(active):
            continue
        overlay[active] = overlay[active] * (1 - alpha) + color_map[name] * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_dark_cluster_mask(brown_black_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Keeps only connected dark components large enough to matter."""
    if brown_black_mask is None:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        brown_black_mask,
        connectivity=8,
    )

    cluster_mask = np.zeros_like(brown_black_mask)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cluster_mask[labels == label_idx] = 255

    return cluster_mask


def save_fresh_debug_artifacts(
    original_image: Image.Image,
    corrected_rgb: np.ndarray,
    crop_rgb: np.ndarray,
    masks: dict,
    result: dict,
    output_root: str = "data/debug_artifacts",
) -> dict:
    """
    Saves debug images for one scan and returns relative/absolute paths.

    Files saved:
    - original.jpg
    - corrected.jpg
    - crop.jpg
    - yellow_mask.png
    - green_mask.png
    - brown_black_mask.png
    - sticker_mask.png, when available
    - dark_cluster_mask.png
    - overlay.jpg
    """
    product_key = result.get("product_key", "unknown")
    result_class = result.get("class", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_id = result.get("inference_id") or str(uuid.uuid4())

    folder = os.path.join(output_root, product_key, f"{ts}_{result_class}_{scan_id[:8]}")
    _ensure_dir(folder)

    original_rgb = np.array(original_image.convert("RGB"))
    corrected_rgb = corrected_rgb if corrected_rgb is not None else original_rgb

    dark_cluster_mask = build_dark_cluster_mask(masks.get("brown_black_mask"), min_area=50)
    masks = {**masks, "dark_cluster_mask": dark_cluster_mask}

    paths = {}

    def add_path(key, filename):
        path = os.path.join(folder, filename)
        paths[key] = path
        return path

    _save_rgb(add_path("original", "original.jpg"), original_rgb)
    _save_rgb(add_path("corrected", "corrected.jpg"), corrected_rgb)
    _save_rgb(add_path("crop", "crop.jpg"), crop_rgb)

    for mask_name in ["yellow_mask", "green_mask", "brown_black_mask", "sticker_mask", "dark_cluster_mask"]:
        mask = masks.get(mask_name)
        if mask is None:
            continue
        _save_mask(add_path(mask_name, f"{mask_name}.png"), mask)

    overlay = create_mask_overlay(crop_rgb, masks)
    _save_rgb(add_path("overlay", "overlay.jpg"), overlay)

    return {
        "debug_saved": True,
        "debug_dir": folder,
        "files": paths,
        "legend": {
            "yellow": "yellow/ripeness mask",
            "green": "green/unripe mask",
            "red": "brown-black defect mask",
            "magenta": "large dark clusters",
            "blue": "sticker/logo excluded area",
        },
    }
