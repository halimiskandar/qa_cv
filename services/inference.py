import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from services.model_registry import get_active_model_config


_MODEL_CACHE = {}
_CONFIG_CACHE = {}


def resize_image(image: Image.Image, longest_side: int = 640) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size

    scale = longest_side / max(w, h)

    if scale >= 1:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return image.resize((new_w, new_h), Image.LANCZOS)


def get_model_config(model_key: str):
    if model_key not in _CONFIG_CACHE:
        _CONFIG_CACHE[model_key] = get_active_model_config(model_key)

    return _CONFIG_CACHE[model_key]


def get_model(model_key: str):
    config = get_model_config(model_key)
    model_path = config["model_path"]

    cache_key = f"{model_key}:{config.get('model_version', 'v1')}:{model_path}"

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = YOLO(model_path)

    return _MODEL_CACHE[cache_key], config


def run_inference(model_type: str, image: Image.Image):
    longest_side = int(os.getenv("DOWNSAMPLED_LONGEST_SIDE", "640"))
    image = resize_image(image, longest_side=longest_side)

    model, config = get_model(model_type)

    if model_type == "banana_ripeness":
        if config["model_type"] == "rule":
            return infer_banana(model, image)

        if config["model_type"] == "classification":
            return infer_banana_classification(
                model,
                image,
                confidence_threshold=config.get("confidence_threshold", 0.75)
            )

    if model_type == "meat_fat_ratio":
        return infer_meat_fat_ratio(model, image)

    if model_type == "fresh_generic":
        return infer_generic_fresh(model, image)

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_reference_card_box(img_rgb):
    """
    MVP assumption:
    Color card is always placed in bottom-left corner of image.
    Adjust these ratios if your card position is different.
    """
    h, w = img_rgb.shape[:2]

    x1 = int(w * 0.03)
    y1 = int(h * 0.72)
    x2 = int(w * 0.33)
    y2 = int(h * 0.97)

    return x1, y1, x2, y2


def split_reference_card_patches(ref_crop):
    """
    Expected layout:
    WHITE | GRAY  | BLACK
    YELLOW| GREEN | BROWN
    """
    h, w = ref_crop.shape[:2]

    patch_boxes = {
        "white":  (0,       0,       w // 3,     h // 2),
        "gray":   (w // 3,  0,       2 * w // 3, h // 2),
        "black":  (2*w//3,  0,       w,          h // 2),
        "yellow": (0,       h // 2,  w // 3,     h),
        "green":  (w // 3,  h // 2,  2 * w // 3, h),
        "brown":  (2*w//3,  h // 2,  w,          h),
    }

    patches = {}

    for name, (x1, y1, x2, y2) in patch_boxes.items():
        patch = ref_crop[y1:y2, x1:x2]

        # use center area only to avoid border/printing edge noise
        ph, pw = patch.shape[:2]
        cx1 = int(pw * 0.25)
        cy1 = int(ph * 0.25)
        cx2 = int(pw * 0.75)
        cy2 = int(ph * 0.75)

        center_patch = patch[cy1:cy2, cx1:cx2]
        patches[name] = center_patch

    return patches


def apply_color_reference_correction(img_rgb, use_reference_card=True):
    """
    Simple per-channel correction using gray patch.
    This stabilizes lighting/color cast before HSV rules.

    Returns:
    corrected_img, reference_debug
    """
    if not use_reference_card:
        return img_rgb, {
            "reference_used": False,
            "reason": "disabled"
        }

    try:
        x1, y1, x2, y2 = get_reference_card_box(img_rgb)
        ref_crop = img_rgb[y1:y2, x1:x2]

        if ref_crop.size == 0:
            return img_rgb, {
                "reference_used": False,
                "reason": "empty_reference_crop"
            }

        patches = split_reference_card_patches(ref_crop)

        expected_rgb = {
            "white":  np.array([255, 255, 255], dtype=np.float32),
            "gray":   np.array([128, 128, 128], dtype=np.float32),
            "black":  np.array([0, 0, 0], dtype=np.float32),
            "yellow": np.array([255, 220, 0], dtype=np.float32),
            "green":  np.array([0, 150, 70], dtype=np.float32),
            "brown":  np.array([120, 70, 30], dtype=np.float32),
        }

        observed_rgb = {}

        for name, patch in patches.items():
            observed_rgb[name] = patch.reshape(-1, 3).mean(axis=0)

        # Main correction uses gray patch.
        # White/black/yellow/green/brown are kept for debugging.
        observed_gray = observed_rgb["gray"]
        expected_gray = expected_rgb["gray"]

        scale = expected_gray / np.maximum(observed_gray, 1)

        # prevent extreme correction if card is badly detected
        scale = np.clip(scale, 0.65, 1.60)

        corrected = img_rgb.astype(np.float32) * scale
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        reference_debug = {
            "reference_used": True,
            "reference_box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "observed_rgb": {
                k: [round(float(v[0]), 2), round(float(v[1]), 2), round(float(v[2]), 2)]
                for k, v in observed_rgb.items()
            },
            "expected_rgb": {
                k: [int(v[0]), int(v[1]), int(v[2])]
                for k, v in expected_rgb.items()
            },
            "rgb_scale": [
                round(float(scale[0]), 4),
                round(float(scale[1]), 4),
                round(float(scale[2]), 4),
            ]
        }

        return corrected, reference_debug

    except Exception as e:
        return img_rgb, {
            "reference_used": False,
            "reason": f"reference_correction_failed: {str(e)}"
        }
    
def remove_logo_from_masks(crop_rgb, yellow_mask, green_mask, brown_black_mask):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    white_sticker = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    green_sticker = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))

    sticker_mask = cv2.bitwise_or(white_sticker, green_sticker)

    kernel = np.ones((9, 9), np.uint8)
    sticker_mask = cv2.morphologyEx(sticker_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    sticker_mask,
    connectivity=8
    )

    cleaned_mask = np.zeros_like(sticker_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 150:
            continue

        aspect_ratio = w / max(h, 1)

        # sticker usually compact
        if 0.5 <= aspect_ratio <= 2.0:
            cleaned_mask[labels == i] = 255

    sticker_mask = cleaned_mask
    sticker_mask = cv2.dilate(sticker_mask, kernel, iterations=2)

    keep_mask = cv2.bitwise_not(sticker_mask)

    yellow_mask = cv2.bitwise_and(yellow_mask, keep_mask)
    green_mask = cv2.bitwise_and(green_mask, keep_mask)
    brown_black_mask = cv2.bitwise_and(brown_black_mask, keep_mask)

    return yellow_mask, green_mask, brown_black_mask, sticker_mask

def get_dark_cluster_metrics(brown_black_mask, total_pixels):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        brown_black_mask,
        connectivity=8
    )

    if num_labels <= 1:
        return {
            "largest_dark_cluster_ratio": 0.0,
            "dark_cluster_count": 0
        }

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    meaningful_areas = areas[areas >= 50]

    if len(meaningful_areas) == 0:
        return {
            "largest_dark_cluster_ratio": 0.0,
            "dark_cluster_count": 0
        }

    largest_area = meaningful_areas.max()

    return {
        "largest_dark_cluster_ratio": float(largest_area / max(total_pixels, 1)),
        "dark_cluster_count": int(len(meaningful_areas))
    }

def infer_banana(model, image: Image.Image):
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "320"))
    use_reference_card = os.getenv("USE_COLOR_REFERENCE_CARD", "true").lower() == "true"

    img = np.array(image.convert("RGB"))

    # 1. Correct lighting/color first
    img, reference_debug = apply_color_reference_correction(
        img,
        use_reference_card=use_reference_card
    )

    image = Image.fromarray(img)

    # 2. YOLO banana detection
    results = model.predict(
        image,
        imgsz=yolo_imgsz,
        conf=0.25,
        verbose=False
    )[0]

    best_box = None
    best_confidence = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        if class_name == "banana" and confidence > best_confidence:
            best_box = box
            best_confidence = confidence

    img_h, img_w = img.shape[:2]

    if best_box is None:
        fallback_used = True
        best_confidence = 0.15

        x1, y1, x2, y2 = 0, 0, img_w, img_h
        crop = img
        coverage_ratio = 1.0
        touches_edge = True
    else:
        fallback_used = False

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        box_w = x2 - x1
        box_h = y2 - y1

        banana_area = box_w * box_h
        image_area = img_w * img_h
        coverage_ratio = banana_area / image_area if image_area > 0 else 0

        margin = 5
        touches_edge = (
            x1 <= margin or
            y1 <= margin or
            x2 >= img_w - margin or
            y2 >= img_h - margin
        )

        crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            "class": "manual_review",
            "confidence": round(best_confidence, 4),
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "invalid_banana_crop",
            "reference_card": reference_debug,
            "quality_metrics": {
                "coverage_ratio": round(float(coverage_ratio), 4),
                "touches_edge": bool(touches_edge),
                "fallback_used": bool(fallback_used),
            }
        }

    # 3. Quality metrics
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    hsv_glare = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    glare_mask = cv2.inRange(
        hsv_glare,
        (0, 0, 245),
        (180, 35, 255)
    )
    glare_ratio = glare_mask.sum() / glare_mask.size

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    edge_gray = cv2.Canny(gray, 80, 160)
    edge_ratio = edge_gray.sum() / 255 / edge_gray.size

    plastic_signal = (
        glare_ratio > 0.008 or
        edge_ratio > 0.08 or
        fallback_used
    )

    plastic_mode = "with_plastic" if plastic_signal else "no_plastic"

    # 4. Color masks
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    if plastic_mode == "with_plastic":
        yellow_mask = cv2.inRange(hsv, (18, 45, 50), (38, 255, 255))

        green_mask_1 = cv2.inRange(hsv, (35, 35, 35), (90, 255, 255))
        green_mask_2 = cv2.inRange(hsv, (25, 15, 40), (50, 220, 245))

        unripe_green_threshold = 0.08
        ready_green_max = 0.05
        minor_brown_max = 0.12
        reject_brown_threshold = 0.20

    else:
        yellow_mask = cv2.inRange(hsv, (18, 50, 50), (38, 255, 255))

        green_mask_1 = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_mask_2 = np.zeros_like(green_mask_1)

        unripe_green_threshold = 0.25
        ready_green_max = 0.20
        minor_brown_max = 0.18
        reject_brown_threshold = 0.24

    green_mask = cv2.bitwise_or(green_mask_1, green_mask_2)

    dark_mask = cv2.inRange(hsv, (0, 20, 0), (180, 255, 70))
    brown_or_darker_mask = cv2.inRange(hsv, (3, 35, 30), (35, 255, 135))
    brown_mask = cv2.inRange(hsv, (5, 40, 35), (30, 255, 120))
    brown_black_mask = cv2.bitwise_or(dark_mask, brown_mask)

    kernel = np.ones((5, 5), np.uint8)
    brown_black_mask = cv2.morphologyEx(brown_black_mask, cv2.MORPH_OPEN, kernel)
    brown_black_mask = cv2.morphologyEx(brown_black_mask, cv2.MORPH_CLOSE, kernel)

    yellow_mask, green_mask, brown_black_mask, sticker_mask = remove_logo_from_masks(
        crop,
        yellow_mask,
        green_mask,
        brown_black_mask
    )

    total_pixels = crop.shape[0] * crop.shape[1]
    sticker_pixels = sticker_mask.sum() / 255
    sticker_ratio = sticker_pixels / total_pixels if total_pixels > 0 else 0

    valid_pixels = max(total_pixels - sticker_pixels, 1)

    yellow_ratio = yellow_mask.sum() / 255 / valid_pixels
    green_ratio_1 = green_mask_1.sum() / 255 / valid_pixels
    green_ratio_2 = green_mask_2.sum() / 255 / valid_pixels
    green_ratio = green_mask.sum() / 255 / valid_pixels
    brown_black_ratio = brown_black_mask.sum() / 255 / valid_pixels

    dark_cluster_metrics = get_dark_cluster_metrics(
    brown_black_mask,
    valid_pixels
    )

    largest_dark_cluster_ratio = dark_cluster_metrics["largest_dark_cluster_ratio"]
    dark_cluster_count = dark_cluster_metrics["dark_cluster_count"]

    if plastic_mode == "with_plastic":
        max_dark_cluster_ratio = 0.015
        max_dark_cluster_count = 5
    else:
        max_dark_cluster_ratio = 0.020
        max_dark_cluster_count = 5

    # 5. Decision logic
    result_class = "manual_review"
    is_accepted = None
    reject_reason = None
    instruction = None
    quality_warning = None

    if touches_edge:
        quality_warning = "banana_touches_edge_result_may_be_less_accurate"

    if best_confidence < 0.20 and not fallback_used:
        reject_reason = "low_yolo_confidence"
        instruction = "Please review manually. Banana was detected with low confidence."

    elif glare_ratio > 0.08:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "too_much_glare"
        instruction = "Please retake photo with less reflection. Avoid flash or direct light."

    elif blur_score < 40:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "photo_too_blurry"
        instruction = "Please retake photo. Hold the camera steady and make sure banana is in focus."

    elif coverage_ratio < 0.25:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "banana_too_small"
        instruction = "Please retake photo closer to the banana."

    elif green_ratio > unripe_green_threshold:
        result_class = "unripe"
        is_accepted = False
        reject_reason = "too_green"

    elif (
        largest_dark_cluster_ratio > max_dark_cluster_ratio
        or (brown_black_ratio > 0.055 and dark_cluster_count >= max_dark_cluster_count)
    ):
        result_class = "reject"
        is_accepted = False
        reject_reason = "visible_dark_bruise_clusters"

    elif brown_black_ratio > reject_brown_threshold:
        result_class = "reject"
        is_accepted = False
        reject_reason = "too_many_dark_spots"

    elif (
        yellow_ratio >= 0.45
        and green_ratio <= ready_green_max
        and brown_black_ratio <= minor_brown_max
    ):
        result_class = "ready_to_send"
        is_accepted = True
        reject_reason = None

    elif yellow_ratio >= 0.35 and green_ratio > 0.05:
        result_class = "almost_ripe"
        is_accepted = None
        reject_reason = "mixed_ripeness"

    else:
        result_class = "manual_review"
        is_accepted = None
        reject_reason = "uncertain_color_quality"
        instruction = "Please review manually. Color quality is uncertain."

    return {
        "class": result_class,
        "confidence": round(best_confidence, 4),
        "is_accepted": is_accepted,
        "needs_manual_review": is_accepted is not True,
        "reject_reason": reject_reason,
        "instruction": instruction,
        "quality_warning": quality_warning,

        "reference_card": reference_debug,

        "packaging": {
            "plastic_detected": plastic_mode == "with_plastic",
            "plastic_mode": plastic_mode,
            "plastic_confidence_signal": {
                "glare_ratio": round(float(glare_ratio), 4),
                "edge_ratio": round(float(edge_ratio), 4),
                "fallback_used": bool(fallback_used)
            }
        },

        "quality_metrics": {
            "coverage_ratio": round(float(coverage_ratio), 4),
            "touches_edge": bool(touches_edge),
            "glare_ratio": round(float(glare_ratio), 4),
            "blur_score": round(float(blur_score), 2),
            "edge_ratio": round(float(edge_ratio), 4),
            "plastic_mode": plastic_mode,
            "fallback_used": bool(fallback_used),
        },

        "color_metrics": {
            "yellow_ratio": round(float(yellow_ratio), 4),
            "green_ratio": round(float(green_ratio), 4),
            "green_ratio_normal": round(float(green_ratio_1), 4),
            "green_ratio_plastic": round(float(green_ratio_2), 4),
            "brown_black_ratio": round(float(brown_black_ratio), 4)
        },
        "object_masks": {
            "sticker_detected": bool(sticker_ratio > 0.01),
            "sticker_ratio": round(float(sticker_ratio), 4)
        },

        "defect_metrics": {
            "largest_dark_cluster_ratio": round(float(largest_dark_cluster_ratio), 4),
            "dark_cluster_count": int(dark_cluster_count),
            "max_dark_cluster_ratio": max_dark_cluster_ratio
        },
        "thresholds": {
            "unripe_green_threshold": unripe_green_threshold,
            "ready_green_max": ready_green_max,
            "minor_brown_max": minor_brown_max,
            "reject_brown_threshold": reject_brown_threshold
        },

        "note": "hybrid model: YOLO banana detection + color-reference correction + plastic-aware OpenCV color rules"
    }

def infer_generic_fresh(model, image: Image.Image):
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "320"))
    use_reference_card = os.getenv("USE_COLOR_REFERENCE_CARD", "true").lower() == "true"

    img = np.array(image.convert("RGB"))

    img, reference_debug = apply_color_reference_correction(
        img,
        use_reference_card=use_reference_card
    )

    image = Image.fromarray(img)

    results = model.predict(
        image,
        imgsz=yolo_imgsz,
        conf=0.20,
        verbose=False
    )[0]

    allowed_classes = {
        "apple",
        "orange",
        "banana",
        "broccoli",
        "carrot"
    }

    best_box = None
    best_confidence = 0
    detected_class = "unknown_fresh"

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        if class_name in allowed_classes and confidence > best_confidence:
            best_box = box
            best_confidence = confidence
            detected_class = class_name

    img_h, img_w = img.shape[:2]

    if best_box is None:
        fallback_used = True
        best_confidence = 0.15
        detected_class = "unknown_fresh"

        h, w = img.shape[:2]

        cx1 = int(w * 0.15)
        cy1 = int(h * 0.15)
        cx2 = int(w * 0.85)
        cy2 = int(h * 0.85)

        crop = img[cy1:cy2, cx1:cx2]
        coverage_ratio = 0.49
        touches_edge = False
    else:
        fallback_used = False

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        crop = img[y1:y2, x1:x2]

        box_w = x2 - x1
        box_h = y2 - y1
        coverage_ratio = (box_w * box_h) / max(img_w * img_h, 1)

        margin = 5
        touches_edge = (
            x1 <= margin or
            y1 <= margin or
            x2 >= img_w - margin or
            y2 >= img_h - margin
        )

    if crop.size == 0:
        return {
            "class": "manual_review",
            "detected_product": detected_class,
            "confidence": round(best_confidence, 4),
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "invalid_crop"
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    hsv_glare = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    glare_mask = cv2.inRange(
        hsv_glare,
        (0, 0, 245),
        (180, 35, 255)
    )

    glare_ratio = glare_mask.sum() / 255 / glare_mask.size
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 65))
    brown_or_darker_mask = cv2.inRange(hsv, (3, 25, 35), (35, 255, 145))

    bruise_mask = cv2.bitwise_or(dark_mask, brown_or_darker_mask)

    kernel = np.ones((5, 5), np.uint8)
    bruise_mask = cv2.morphologyEx(bruise_mask, cv2.MORPH_OPEN, kernel)
    bruise_mask = cv2.morphologyEx(bruise_mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = crop.shape[0] * crop.shape[1]
    bruise_ratio = bruise_mask.sum() / 255 / max(total_pixels, 1)

    cluster_metrics = get_dark_cluster_metrics(
        bruise_mask,
        total_pixels
    )

    largest_bruise_cluster_ratio = cluster_metrics["largest_dark_cluster_ratio"]
    bruise_cluster_count = cluster_metrics["dark_cluster_count"]

    max_bruise_ratio = 0.10
    max_bruise_cluster_ratio = 0.018
    max_bruise_cluster_count = 5

    result_class = "ready_to_send"
    is_accepted = True
    reject_reason = None
    instruction = None
    quality_warning = None

    if touches_edge:
        quality_warning = "item_touches_edge_result_may_be_less_accurate"

    if blur_score < 40:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "photo_too_blurry"
        instruction = "Please retake photo. Hold camera steady."

    elif glare_ratio > 0.08:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "too_much_glare"
        instruction = "Please retake photo with less reflection."

    elif coverage_ratio < 0.20:
        result_class = "retake_photo"
        is_accepted = False
        reject_reason = "item_too_small"
        instruction = "Please retake photo closer to the item."

    elif (
        largest_bruise_cluster_ratio > max_bruise_cluster_ratio
        or bruise_ratio > max_bruise_ratio
        or (bruise_ratio > 0.045 and bruise_cluster_count >= max_bruise_cluster_count)
    ):
        result_class = "reject"
        is_accepted = False
        reject_reason = "visible_bruise_or_dark_damage"

    return {
        "class": result_class,
        "detected_product": detected_class,
        "confidence": round(best_confidence, 4),
        "is_accepted": is_accepted,
        "needs_manual_review": is_accepted is not True,
        "reject_reason": reject_reason,
        "instruction": instruction,
        "quality_warning": quality_warning,

        "reference_card": reference_debug,

        "quality_metrics": {
            "coverage_ratio": round(float(coverage_ratio), 4),
            "touches_edge": bool(touches_edge),
            "glare_ratio": round(float(glare_ratio), 4),
            "blur_score": round(float(blur_score), 2),
            "fallback_used": bool(fallback_used)
        },

        "defect_metrics": {
            "bruise_ratio": round(float(bruise_ratio), 4),
            "largest_bruise_cluster_ratio": round(float(largest_bruise_cluster_ratio), 4),
            "bruise_cluster_count": int(bruise_cluster_count),
            "max_bruise_ratio": max_bruise_ratio,
            "max_bruise_cluster_ratio": max_bruise_cluster_ratio
        },

        "thresholds": {
            "max_bruise_ratio": max_bruise_ratio,
            "max_bruise_cluster_ratio": max_bruise_cluster_ratio,
            "max_bruise_cluster_count": max_bruise_cluster_count
        },

        "note": "generic fresh QA: YOLO detection + color-reference correction + darker bruise cluster detection"
    }
    
def infer_banana_classification(model, image: Image.Image, confidence_threshold=0.75):
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "320"))

    results = model.predict(
        image,
        imgsz=yolo_imgsz,
        conf=0.25,
        verbose=False
    )[0]

    probs = results.probs

    if probs is None:
        return {
            "class": "manual_review",
            "confidence": 0,
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "no_classification_probs"
        }

    top_idx = int(probs.top1)
    confidence = float(probs.top1conf)
    class_name = model.names[top_idx]

    if confidence < confidence_threshold:
        return {
            "class": "manual_review",
            "raw_class": class_name,
            "confidence": round(confidence, 4),
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "low_model_confidence"
        }

    reject_classes = {
        "unripe",
        "overripe_reject",
        "black_spots_reject",
        "loose_stalk_reject",
        "bruised_reject",
        "stalk_issue_reject"
    }

    return {
        "class": class_name,
        "confidence": round(confidence, 4),
        "is_accepted": class_name not in reject_classes,
        "needs_manual_review": False,
        "reject_reason": class_name if class_name in reject_classes else None
    }


def infer_meat_fat_ratio(model, image: Image.Image):
    img = np.array(image.convert("RGB"))
    img_h, img_w = img.shape[:2]
    

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hsv_glare = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # true glare:
    # very bright + very low saturation
    glare_mask = cv2.inRange(
        hsv_glare,
        (0, 0, 245),
        (180, 35, 255)
    )

    glare_ratio = glare_mask.sum() / 255 / glare_mask.size

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if glare_ratio > 0.10:
        return {
            "class": "retake_photo",
            "fat_ratio_pct": None,
            "meat_ratio_pct": None,
            "is_accepted": False,
            "needs_manual_review": True,
            "reject_reason": "too_much_glare",
            "instruction": "Please retake photo with less reflection. Avoid flash or direct light.",
            "quality_metrics": {
                "glare_ratio": round(float(glare_ratio), 4)
            },
            "note": "meat MVP: HSV fat/meat ratio rule"
        }

    if blur_score < 40:
        return {
            "class": "retake_photo",
            "fat_ratio_pct": None,
            "meat_ratio_pct": None,
            "is_accepted": False,
            "needs_manual_review": True,
            "reject_reason": "photo_too_blurry",
            "instruction": "Please retake photo. Hold camera steady and make sure meat is in focus.",
            "quality_metrics": {
                "glare_ratio": round(float(glare_ratio), 4),
                "blur_score": round(float(blur_score), 2)
            },
            "note": "meat MVP: HSV fat/meat ratio rule"
        }

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    fat_mask = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))

    red_mask_1 = cv2.inRange(hsv, (0, 60, 40), (12, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 60, 40), (180, 255, 255))
    meat_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
    brown_mask = cv2.inRange(hsv, (8, 40, 30), (30, 255, 130))
    spoil_mask = cv2.bitwise_or(dark_mask, brown_mask)

    kernel = np.ones((5, 5), np.uint8)

    fat_mask = cv2.morphologyEx(fat_mask, cv2.MORPH_OPEN, kernel)
    meat_mask = cv2.morphologyEx(meat_mask, cv2.MORPH_OPEN, kernel)
    spoil_mask = cv2.morphologyEx(spoil_mask, cv2.MORPH_OPEN, kernel)

    fat_pixels = fat_mask.sum() / 255
    meat_pixels = meat_mask.sum() / 255
    spoil_pixels = spoil_mask.sum() / 255

    valid_pixels = fat_pixels + meat_pixels
    total_pixels = img_h * img_w

    if valid_pixels <= 0:
        return {
            "class": "manual_review",
            "fat_ratio_pct": None,
            "meat_ratio_pct": None,
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "no_meat_or_fat_region_detected",
            "quality_metrics": {
                "glare_ratio": round(float(glare_ratio), 4),
                "blur_score": round(float(blur_score), 2)
            },
            "note": "meat MVP: HSV fat/meat ratio rule"
        }

    fat_ratio_pct = fat_pixels / valid_pixels * 100
    meat_ratio_pct = meat_pixels / valid_pixels * 100
    spoil_ratio_pct = spoil_pixels / total_pixels * 100
    valid_area_ratio = valid_pixels / total_pixels

    result_class = "acceptable"
    is_accepted = True
    needs_manual_review = False
    reject_reason = None
    instruction = None

    if valid_area_ratio < 0.20:
        result_class = "manual_review"
        is_accepted = None
        needs_manual_review = True
        reject_reason = "meat_region_too_small_or_background_confusing"
        instruction = "Please retake photo closer to the meat with less background."

    elif spoil_ratio_pct > 10:
        result_class = "spoiled_or_bad_color"
        is_accepted = False
        needs_manual_review = True
        reject_reason = "possible_spoilage_or_bad_color"
        instruction = "Please review manually. Meat color appears too dark/brown."

    elif fat_ratio_pct > 35:
        result_class = "too_fatty"
        is_accepted = False
        needs_manual_review = False
        reject_reason = "fat_ratio_too_high"

    elif fat_ratio_pct < 8:
        result_class = "too_lean"
        is_accepted = False
        needs_manual_review = False
        reject_reason = "fat_ratio_too_low"

    return {
        "class": result_class,
        "fat_ratio_pct": round(float(fat_ratio_pct), 2),
        "meat_ratio_pct": round(float(meat_ratio_pct), 2),
        "is_accepted": is_accepted,
        "needs_manual_review": needs_manual_review,
        "reject_reason": reject_reason,
        "instruction": instruction,
        "quality_metrics": {
            "glare_ratio": round(float(glare_ratio), 4),
            "blur_score": round(float(blur_score), 2),
            "valid_area_ratio": round(float(valid_area_ratio), 4),
            "spoil_ratio_pct": round(float(spoil_ratio_pct), 2)
        },
        "color_metrics": {
            "fat_pixels": int(fat_pixels),
            "meat_pixels": int(meat_pixels),
            "spoil_pixels": int(spoil_pixels)
        },
        "thresholds": {
            "max_fat_ratio_pct": 35,
            "min_fat_ratio_pct": 8,
            "max_spoil_ratio_pct": 10
        },
        "note": "meat MVP: HSV segmentation for red meat vs white fat"
    }