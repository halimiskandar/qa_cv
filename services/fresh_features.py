import cv2
import numpy as np

def remove_banana_tip_darkness(crop_rgb, brown_black_mask, product_cfg):
    if product_cfg.get("decision_profile") != "banana":
        return brown_black_mask

    h, w = brown_black_mask.shape

    tip_mask = np.zeros_like(brown_black_mask)

    # ignore left/right 18% because banana tips/stem are often naturally dark
    tip_width = int(w * 0.18)

    tip_mask[:, :tip_width] = 255
    tip_mask[:, w - tip_width:] = 255

    # also ignore top-left stem area
    tip_mask[:int(h * 0.35), :int(w * 0.30)] = 255

    keep_mask = cv2.bitwise_not(tip_mask)

    return cv2.bitwise_and(brown_black_mask, keep_mask)

def compute_quality_metrics(
    crop_rgb,
    coverage_ratio,
    touches_edge,
    fallback_used,
    product_cfg
):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    glare_ratio = (gray > 245).sum() / gray.size
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    blur_score = laplacian.var()

    surface_cfg = product_cfg.get("surface", {})

    if surface_cfg.get("texture_detector", False):
        texture_penalty = np.mean(np.abs(laplacian))

        blur_score = blur_score + (texture_penalty * 1.5)
    edge_gray = cv2.Canny(gray, 80, 160)
    edge_ratio = edge_gray.sum() / 255 / edge_gray.size
    plastic_signal = glare_ratio > 0.008 or edge_ratio > 0.08 or fallback_used
    plastic_mode = "with_plastic" if plastic_signal else "no_plastic"
    return {
        "coverage_ratio": float(coverage_ratio),
        "touches_edge": bool(touches_edge),
        "glare_ratio": float(glare_ratio),
        "blur_score": float(blur_score),
        "edge_ratio": float(edge_ratio),
        "plastic_mode": plastic_mode,
        "fallback_used": bool(fallback_used),
    }


def build_hsv_masks(crop_rgb, product_cfg, plastic_mode):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    plastic_glare_mask = ((saturation < 45) & (value > 170)).astype(np.uint8) * 255
    valid_color_mask = cv2.bitwise_not(plastic_glare_mask)


    yellow_leaf_mask = cv2.inRange(hsv, (18, 40, 40), (40, 255, 255))
    brown_leaf_mask = cv2.inRange(hsv, (5, 35, 20), (30, 255, 120))
    black_leaf_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
    cfg = product_cfg["plastic" if plastic_mode == "with_plastic" else "no_plastic"]

    yellow_range = cfg.get("yellow_hsv", ((18, 45, 50), (38, 255, 255)))
    green_range_1 = cfg.get("green_hsv_1", ((35, 40, 40), (90, 255, 255)))
    green_range_2 = cfg.get("green_hsv_2")

    yellow_mask = cv2.inRange(hsv, yellow_range[0], yellow_range[1])
    green_mask_1 = cv2.inRange(hsv, green_range_1[0], green_range_1[1])
    green_mask_2 = cv2.inRange(hsv, green_range_2[0], green_range_2[1]) if green_range_2 else np.zeros_like(green_mask_1)
    green_mask = cv2.bitwise_or(green_mask_1, green_mask_2)
    green_mask = cv2.bitwise_and(green_mask, valid_color_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, valid_color_mask)

    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 55))
    brown_mask = cv2.inRange(hsv, (5, 40, 35), (30, 255, 120))

    brown_black_mask = cv2.bitwise_or(dark_mask, brown_mask)

    # apple protection
    if product_cfg.get("display_name", "").lower() == "apple":
        red_mask_1 = cv2.inRange(hsv, (0, 80, 40), (12, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (165, 80, 40), (180, 255, 255))

        healthy_red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )

        brown_black_mask = cv2.bitwise_and(
            brown_black_mask,
            cv2.bitwise_not(healthy_red_mask)
        )

    surface_cfg = product_cfg.get("surface", {})

    kernel_size = surface_cfg.get("morph_kernel", 5)

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )
    brown_black_mask = cv2.morphologyEx(brown_black_mask, cv2.MORPH_OPEN, kernel)
    brown_black_mask = cv2.morphologyEx(brown_black_mask, cv2.MORPH_CLOSE, kernel)

    return (
    yellow_mask,
    green_mask_1,
    green_mask_2,
    green_mask,
    brown_black_mask,
    yellow_leaf_mask,
    brown_leaf_mask,
    black_leaf_mask,
)


def remove_logo_from_masks(crop_rgb, yellow_mask, green_mask, brown_black_mask):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    white_sticker = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    green_sticker = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    sticker_mask = cv2.bitwise_or(white_sticker, green_sticker)

    kernel = np.ones((9, 9), np.uint8)
    sticker_mask = cv2.morphologyEx(sticker_mask, cv2.MORPH_CLOSE, kernel)
    sticker_mask = cv2.dilate(sticker_mask, kernel, iterations=2)
    keep_mask = cv2.bitwise_not(sticker_mask)

    return (
        cv2.bitwise_and(yellow_mask, keep_mask),
        cv2.bitwise_and(green_mask, keep_mask),
        cv2.bitwise_and(brown_black_mask, keep_mask),
        sticker_mask,
    )


def get_dark_cluster_metrics(brown_black_mask, valid_pixels):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(brown_black_mask, connectivity=8)
    if num_labels <= 1:
        return {"largest_dark_cluster_ratio": 0.0, "dark_cluster_count": 0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    meaningful = areas[areas >= 50]
    if len(meaningful) == 0:
        return {"largest_dark_cluster_ratio": 0.0, "dark_cluster_count": 0}
    return {
        "largest_dark_cluster_ratio": float(meaningful.max() / max(valid_pixels, 1)),
        "dark_cluster_count": int(len(meaningful)),
    }

def build_foreground_mask(crop_rgb, product_cfg):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    product_name = product_cfg.get("display_name", "").lower()

    if product_name == "orange":
        # orange has high saturation, exclude table/shadow/background
        mask = cv2.inRange(hsv, (5, 45, 45), (35, 255, 255))

    elif product_name == "apple":
        # red/green/yellow apples
        red1 = cv2.inRange(hsv, (0, 45, 35), (15, 255, 255))
        red2 = cv2.inRange(hsv, (160, 45, 35), (180, 255, 255))
        green = cv2.inRange(hsv, (35, 35, 35), (90, 255, 255))
        yellow = cv2.inRange(hsv, (15, 35, 35), (40, 255, 255))
        mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(green, yellow))

    else:
        # generic foreground
        sat_mask = cv2.inRange(hsv, (0, 35, 35), (180, 255, 255))
        val_mask = cv2.inRange(hsv, (0, 0, 45), (180, 255, 255))
        mask = cv2.bitwise_and(sat_mask, val_mask)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # keep largest connected component only
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.ones(crop_rgb.shape[:2], dtype=np.uint8) * 255

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_idx] = 255

    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    return clean_mask

def compute_color_features(crop_rgb, product_cfg, plastic_mode, return_debug_masks=False):
    (
    yellow_mask,
    green_mask_1,
    green_mask_2,
    green_mask,
    brown_black_mask,
    yellow_leaf_mask,
    brown_leaf_mask,
    black_leaf_mask,
    ) = build_hsv_masks(crop_rgb, product_cfg, plastic_mode)

    sticker_mask = np.zeros_like(yellow_mask)
    if product_cfg.get("remove_sticker", False):
        candidate_yellow, candidate_green, candidate_brown, candidate_sticker = remove_logo_from_masks(
            crop_rgb, yellow_mask, green_mask, brown_black_mask
        )

        total_pixels_tmp = crop_rgb.shape[0] * crop_rgb.shape[1]
        candidate_sticker_ratio = candidate_sticker.sum() / 255 / max(total_pixels_tmp, 1)

        if candidate_sticker_ratio <= 0.20:
            yellow_mask = candidate_yellow
            green_mask = candidate_green
            brown_black_mask = candidate_brown
            sticker_mask = candidate_sticker

    foreground_mask = build_foreground_mask(crop_rgb, product_cfg)

    yellow_mask = cv2.bitwise_and(yellow_mask, foreground_mask)
    green_mask = cv2.bitwise_and(green_mask, foreground_mask)
    green_mask_1 = cv2.bitwise_and(green_mask_1, foreground_mask)
    green_mask_2 = cv2.bitwise_and(green_mask_2, foreground_mask)
    brown_black_mask = cv2.bitwise_and(brown_black_mask, foreground_mask)
    brown_black_mask = remove_banana_tip_darkness(
    crop_rgb,
    brown_black_mask,
    product_cfg
)
    yellow_leaf_mask = cv2.bitwise_and(yellow_leaf_mask, foreground_mask)
    brown_leaf_mask = cv2.bitwise_and(brown_leaf_mask, foreground_mask)
    black_leaf_mask = cv2.bitwise_and(black_leaf_mask, foreground_mask)

    total_pixels = crop_rgb.shape[0] * crop_rgb.shape[1]
    sticker_pixels = sticker_mask.sum() / 255
    sticker_ratio = sticker_pixels / total_pixels if total_pixels > 0 else 0
    foreground_pixels = foreground_mask.sum() / 255
    valid_pixels = max(foreground_pixels - sticker_pixels, 1)

    yellow_leaf_ratio = yellow_leaf_mask.sum() / 255 / valid_pixels
    brown_leaf_ratio = brown_leaf_mask.sum() / 255 / valid_pixels
    black_leaf_ratio = black_leaf_mask.sum() / 255 / valid_pixels
    yellow_ratio = yellow_mask.sum() / 255 / valid_pixels
    green_ratio = green_mask.sum() / 255 / valid_pixels
    brown_black_ratio = brown_black_mask.sum() / 255 / valid_pixels

    cluster = get_dark_cluster_metrics(brown_black_mask, valid_pixels)

    result = {
        "color_metrics": {
            "yellow_ratio": float(yellow_ratio),
            "green_ratio": float(green_ratio),
            "green_ratio_normal": float(green_mask_1.sum() / 255 / valid_pixels),
            "green_ratio_plastic": float(green_mask_2.sum() / 255 / valid_pixels),
            "brown_black_ratio": float(brown_black_ratio),
            "yellow_leaf_ratio": float(yellow_leaf_ratio),
            "brown_leaf_ratio": float(brown_leaf_ratio),
            "black_leaf_ratio": float(black_leaf_ratio),
        },
        "object_masks": {
            "sticker_detected": bool(sticker_ratio > 0.01),
            "sticker_ratio": float(sticker_ratio),
        },
        "defect_metrics": cluster,
    }

    if return_debug_masks:
        result["debug_masks"] = {
            "yellow_mask": yellow_mask,
            "green_mask": green_mask,
            "green_mask_1": green_mask_1,
            "green_mask_2": green_mask_2,
            "brown_black_mask": brown_black_mask,
            "sticker_mask": sticker_mask,
        }

    return result
