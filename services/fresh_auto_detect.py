"""Auto-detection router for Fresh QA.

Goal:
- Let YOLO choose the fresh item when possible.
- If the item is known, use a product-specific profile.
- If the item is new/unknown, use a generic fresh profile instead of failing.
"""

import os
from typing import Optional, Tuple

from PIL import Image

from services.fresh_config import (
    DETECTOR_CLASS_TO_PRODUCT_KEY,
    FRESH_PRODUCT_CONFIG,
    PRODUCT_NAME_KEYWORDS,
)
from services.fresh_inference import infer_fresh_product
from services.inference import get_model, resize_image


DEFAULT_AUTO_MODEL_KEY = "banana_ripeness"
DEFAULT_UNKNOWN_PRODUCT_KEY = "generic_fresh"


def product_key_from_name(product_name: Optional[str]) -> Optional[str]:
    if not product_name:
        return None

    text = product_name.lower().strip()
    for product_key, keywords in PRODUCT_NAME_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return product_key

    return None


def detect_fresh_item_from_yolo(
    model_key: str,
    image: Image.Image,
    confidence_threshold: float = 0.20,
) -> dict:
    """Run YOLO once and map the best detected fresh class to a QA profile."""
    model, _ = get_model(model_key)
    yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "320"))
    image = resize_image(image, longest_side=int(os.getenv("DOWNSAMPLED_LONGEST_SIDE", "640")))

    results = model.predict(image, imgsz=yolo_imgsz, conf=confidence_threshold, verbose=False)[0]

    best = {
        "detected": False,
        "detected_class": None,
        "detected_confidence": 0.0,
        "product_key": DEFAULT_UNKNOWN_PRODUCT_KEY,
        "route_source": "default_generic_fresh",
        "all_fresh_candidates": [],
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        mapped_product = DETECTOR_CLASS_TO_PRODUCT_KEY.get(class_name)
        if not mapped_product:
            continue

        candidate = {
            "detected_class": class_name,
            "confidence": round(confidence, 4),
            "product_key": mapped_product,
        }
        best["all_fresh_candidates"].append(candidate)

        if confidence > best["detected_confidence"]:
            best.update({
                "detected": True,
                "detected_class": class_name,
                "detected_confidence": confidence,
                "product_key": mapped_product,
                "route_source": "yolo_detected_class",
            })

    return best


def resolve_product_key(
    image: Image.Image,
    model_key: str = DEFAULT_AUTO_MODEL_KEY,
    product_key: Optional[str] = None,
    product_name: Optional[str] = None,
    allow_auto_detect: bool = True,
) -> Tuple[str, dict]:
    """Resolve final product_key using explicit input, product_name, YOLO, then generic fallback."""
    if product_key and product_key in FRESH_PRODUCT_CONFIG:
        return product_key, {
            "resolved_product_key": product_key,
            "route_source": "explicit_product_key",
            "auto_detected": False,
        }

    name_key = product_key_from_name(product_name)
    if name_key and name_key in FRESH_PRODUCT_CONFIG:
        return name_key, {
            "resolved_product_key": name_key,
            "route_source": "product_name_keyword",
            "auto_detected": False,
            "product_name": product_name,
        }

    if allow_auto_detect:
        detection = detect_fresh_item_from_yolo(model_key=model_key, image=image)
        detected_key = detection.get("product_key") or DEFAULT_UNKNOWN_PRODUCT_KEY
        if detected_key not in FRESH_PRODUCT_CONFIG:
            detected_key = DEFAULT_UNKNOWN_PRODUCT_KEY
        return detected_key, {
            **detection,
            "resolved_product_key": detected_key,
            "auto_detected": detection.get("detected", False),
        }

    return DEFAULT_UNKNOWN_PRODUCT_KEY, {
        "resolved_product_key": DEFAULT_UNKNOWN_PRODUCT_KEY,
        "route_source": "auto_disabled_generic_fresh",
        "auto_detected": False,
    }


def infer_auto_fresh_product(
    image: Image.Image,
    model_key: str = DEFAULT_AUTO_MODEL_KEY,
    product_key: Optional[str] = None,
    product_name: Optional[str] = None,
    save_debug: bool = None,
    debug_output_root: str = "data/debug_artifacts",
    allow_auto_detect: bool = True,
) -> dict:
    """Auto-detect the product, then run the generic fresh QA pipeline."""
    resolved_key, routing = resolve_product_key(
        image=image,
        model_key=model_key,
        product_key=product_key,
        product_name=product_name,
        allow_auto_detect=allow_auto_detect,
    )

    result = infer_fresh_product(
        model_key=model_key,
        image=image,
        product_key=resolved_key,
        save_debug=save_debug,
        debug_output_root=debug_output_root,
    )

    result["routing"] = routing
    result["auto_detected_product_key"] = resolved_key
    result["product_profile"] = {
        "product_key": resolved_key,
        "display_name": FRESH_PRODUCT_CONFIG[resolved_key].get("display_name", resolved_key),
        "decision_profile": FRESH_PRODUCT_CONFIG[resolved_key].get("decision_profile"),
        "generic_family": FRESH_PRODUCT_CONFIG[resolved_key].get("generic_family"),
    }

    if routing.get("route_source") == "default_generic_fresh":
        result["quality_warning"] = result.get("quality_warning") or "unknown_item_using_generic_fresh_profile"

    return result
