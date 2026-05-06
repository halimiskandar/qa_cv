import os
import cv2
import numpy as np
from PIL import Image

from services.fresh_config import FRESH_PRODUCT_CONFIG
from services.fresh_features import compute_quality_metrics, compute_color_features
from services.fresh_rules import decide_fresh_quality
from services.inference import get_model, resize_image, apply_color_reference_correction, infer_meat_fat_ratio
from services.debug_artifacts import save_fresh_debug_artifacts
from services.fallback_fruit_classifier import classify_with_fallback
from services.segmentation import segment_fruit, crop_to_mask


def _find_best_object_box(model, image, detector_classes):
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "320"))
    results = model.predict(image, imgsz=yolo_imgsz, conf=0.25, verbose=False)[0]

    best_box = None
    best_confidence = 0.0
    best_class_name = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        if class_name in detector_classes and confidence > best_confidence:
            best_box = box
            best_confidence = confidence
            best_class_name = class_name

    return best_box, best_confidence, best_class_name


def _crop_from_box(img_rgb, box):
    img_h, img_w = img_rgb.shape[:2]

    if box is None:
        img_h, img_w = img_rgb.shape[:2]

        return img_rgb, {
            "fallback_used": True,
            "confidence": 0.15,
            "detected_class": None,
            "coverage_ratio": 1.0,
            "touches_edge": True,
            "box": {
                "x1": 0,
                "y1": 0,
                "x2": int(img_w),
                "y2": int(img_h),
            },
        }

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    box_w, box_h = x2 - x1, y2 - y1
    coverage_ratio = (box_w * box_h) / (img_w * img_h) if img_w * img_h > 0 else 0
    margin = 5
    touches_edge = x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin

    return img_rgb[y1:y2, x1:x2], {
        "fallback_used": False,
        "coverage_ratio": float(coverage_ratio),
        "touches_edge": bool(touches_edge),
        "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
    }


def infer_fresh_product(
    model_key: str,
    image: Image.Image,
    product_key: str,
    save_debug: bool = None,
    debug_output_root: str = "data/debug_artifacts",
):
    """Generic fresh QA pipeline for banana/apple/tomato/leafy_veg/etc.

    Set save_debug=True, or environment SAVE_DEBUG_ARTIFACTS=true,
    to save crop/mask/overlay images for demos and QA review.
    """
    if product_key == "meat":
        model, _ = get_model("meat_fat_ratio")
        return infer_meat_fat_ratio(model, image)

    if product_key not in FRESH_PRODUCT_CONFIG:
        raise ValueError(f"Unsupported product_key: {product_key}")

    product_cfg = FRESH_PRODUCT_CONFIG[product_key]
    model, _ = get_model(model_key)

    longest_side = int(os.getenv("DOWNSAMPLED_LONGEST_SIDE", "640"))
    image = resize_image(image, longest_side=longest_side)
    original_image = image.copy()
    img = np.array(image.convert("RGB"))

    if product_cfg.get("use_reference_card", True):
        img, reference_debug = apply_color_reference_correction(img, use_reference_card=True)
    else:
        reference_debug = {"reference_used": False, "reason": "disabled_for_product"}

    corrected_image = Image.fromarray(img)
    best_box, confidence, detected_class = _find_best_object_box(
        model,
        corrected_image,
        product_cfg.get("detector_classes", []),
    )

    fallback_result = None

    if product_key == "banana":
        if detected_class is None:
            detected_class = "banana"
            confidence = 0.15

    elif (
        detected_class is None
        or confidence < 0.50
    ):
        fallback_result = classify_with_fallback(image)

        detected_class = fallback_result["product_key"]
        confidence = fallback_result["confidence"]

    # Try YOLO segmentation first
    seg = segment_fruit(
        img,
        expected_class=None,
        conf=0.25
    )

    if seg is not None:
        crop, crop_mask = crop_to_mask(img, seg["mask"])
        crop = crop.copy()
        crop[crop_mask == 0] = [255, 255, 255]

        crop_meta = {
            "fallback_used": False,
            "confidence": seg["confidence"],
            "detected_class": seg.get("detected_class") or seg.get("class_name"),
            "coverage_ratio": float((seg["mask"] > 0).sum() / (img.shape[0] * img.shape[1])),
            "touches_edge": False,
            "segmentation_used": True,
        }

        confidence = seg["confidence"]
        detected_class = seg["detected_class"]

    else:
        # fallback to old YOLO box crop
        crop, crop_meta = _crop_from_box(img, best_box)
        crop_meta["segmentation_used"] = False
    if crop.size == 0:
        return {
            "product_key": product_key,
            "class": "manual_review",
            "confidence": round(float(confidence), 4),
            "is_accepted": None,
            "needs_manual_review": True,
            "reject_reason": "invalid_item_crop",
            "reference_card": reference_debug,
        }

    quality = compute_quality_metrics(
        crop,
        coverage_ratio=crop_meta["coverage_ratio"],
        touches_edge=crop_meta["touches_edge"],
        fallback_used=crop_meta["fallback_used"],
        product_cfg=product_cfg,
    )
    if save_debug is None:
        save_debug = os.getenv("SAVE_DEBUG_ARTIFACTS", "false").lower() == "true"

    color_bundle = compute_color_features(
        crop,
        product_cfg,
        quality["plastic_mode"],
        return_debug_masks=save_debug,
    )
    decision = decide_fresh_quality(
        product_key,
        product_cfg,
        quality,
        color_bundle["color_metrics"],
        color_bundle["defect_metrics"],
    )

    result_payload = {
        "product_key": product_key,
        "detected_class": detected_class,
        "class": decision["class"],
        "confidence": round(float(confidence or 0.15), 4),
        "is_accepted": decision["is_accepted"],
        "needs_manual_review": decision["needs_manual_review"],
        "reject_reason": decision["reject_reason"],
        "instruction": decision["instruction"],
        "quality_warning": decision["quality_warning"],
        "reference_card": reference_debug,
        "packaging": {
            "plastic_detected": quality["plastic_mode"] == "with_plastic",
            "plastic_mode": quality["plastic_mode"],
            "plastic_confidence_signal": {
                "glare_ratio": round(float(quality["glare_ratio"]), 4),
                "edge_ratio": round(float(quality["edge_ratio"]), 4),
                "fallback_used": bool(quality["fallback_used"]),
            },
        },
        "quality_metrics": {
            "coverage_ratio": round(float(quality["coverage_ratio"]), 4),
            "touches_edge": bool(quality["touches_edge"]),
            "glare_ratio": round(float(quality["glare_ratio"]), 4),
            "blur_score": round(float(quality["blur_score"]), 2),
            "edge_ratio": round(float(quality["edge_ratio"]), 4),
            "plastic_mode": quality["plastic_mode"],
            "fallback_used": bool(quality["fallback_used"]),
        },
        "color_metrics": {k: round(float(v), 4) for k, v in color_bundle["color_metrics"].items()},
        "object_masks": {
            "sticker_detected": bool(color_bundle["object_masks"]["sticker_detected"]),
            "sticker_ratio": round(float(color_bundle["object_masks"]["sticker_ratio"]), 4),
        },
        "defect_metrics": {
            "largest_dark_cluster_ratio": round(float(color_bundle["defect_metrics"]["largest_dark_cluster_ratio"]), 4),
            "dark_cluster_count": int(color_bundle["defect_metrics"]["dark_cluster_count"]),
            "max_dark_cluster_ratio": decision["thresholds"].get("max_dark_cluster_ratio"),
            "max_dark_cluster_count": decision["thresholds"].get("max_dark_cluster_count"),
        },
        "fallback_classifier": fallback_result,
        "thresholds": decision["thresholds"],
        "crop": crop_meta,
        "note": "generic fresh QA: YOLO product detection + shared color/defect features + product-specific config rules",
    }

    if save_debug:
        debug_artifacts = save_fresh_debug_artifacts(
            original_image=original_image,
            corrected_rgb=img,
            crop_rgb=crop,
            masks=color_bundle.get("debug_masks", {}),
            result=result_payload,
            output_root=debug_output_root,
        )
        result_payload["debug_artifacts"] = debug_artifacts

    return result_payload
