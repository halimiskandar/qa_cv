import cv2
import numpy as np
from ultralytics import YOLO

SEG_MODEL = YOLO("models/fruit_segmenter.pt")


def clean_segmentation_mask(mask):
    mask = (mask > 127).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    h, w = mask.shape
    image_area = h * w

    clean = np.zeros_like(mask)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]

        if area < image_area * 0.01:
            continue

        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        bw = stats[idx, cv2.CC_STAT_WIDTH]
        bh = stats[idx, cv2.CC_STAT_HEIGHT]

        aspect = max(bw, bh) / max(min(bw, bh), 1)

        if aspect > 8:
            continue

        clean[labels == idx] = 255

    if clean.sum() == 0:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean[labels == largest_idx] = 255

    return clean


def segment_fruit(
    image_rgb,
    expected_class=None,
    conf=0.20
):
    """
    Run YOLO segmentation and return best mask.
    """

    results = SEG_MODEL.predict(
        image_rgb,
        conf=conf,
        verbose=False
    )

    if not results:
        return None

    r = results[0]

    if r.masks is None:
        return None

    best = None
    best_area = 0
    best_confidence = 0.0
    best_class_name = None

    names = r.names

    for i, mask_tensor in enumerate(r.masks.data):
        cls_id = int(r.boxes.cls[i])
        cls_name = names[cls_id]
        confidence = float(r.boxes.conf[i])

        if expected_class and cls_name != expected_class:
            continue

        mask = mask_tensor.cpu().numpy()
        area = mask.sum()

        if area > best_area:
            best_area = area
            best = mask
            best_confidence = confidence
            best_class_name = cls_name

    if best is None:
        return None

    mask = (best * 255).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mask = clean_segmentation_mask(mask)
    return {
        "mask": mask,
        "confidence": best_confidence,
        "detected_class": best_class_name,
        "class_name": best_class_name,
    }


def crop_to_mask(image_rgb, mask):
    """
    Crop image tightly around mask.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image_rgb, mask

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    cropped = image_rgb[y1:y2, x1:x2]

    cropped_mask = mask[y1:y2, x1:x2]

    fruit_only = cv2.bitwise_and(
        cropped,
        cropped,
        mask=cropped_mask
    )

    return fruit_only, cropped_mask