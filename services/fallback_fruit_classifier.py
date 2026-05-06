from pathlib import Path
from ultralytics import YOLO
from PIL import Image

_FALLBACK_MODEL = None
MODEL_PATH = Path("models/fruits360_classifier.pt")


def get_fallback_model():
    global _FALLBACK_MODEL

    if not MODEL_PATH.exists():
        return None

    if _FALLBACK_MODEL is None:
        _FALLBACK_MODEL = YOLO(str(MODEL_PATH))

    return _FALLBACK_MODEL


FRUIT_CLASS_MAP = {
    "banana": "banana",
    "orange": "generic_fruit",
    "apple": "apple",
    "pear": "asian_pear",
    "asian pear": "asian_pear",
    "tomato": "tomato",
}


def classify_with_fallback(image: Image.Image):
    model = get_fallback_model()

    if model is None:
        return {
            "product_key": "generic_fruit",
            "confidence": 0.0,
            "raw_class": None,
            "source": "fallback_model_missing"
        }

    result = model.predict(
        image,
        imgsz=224,
        verbose=False
    )[0]

    probs = result.probs

    if probs is None:
        return {
            "product_key": "generic_fruit",
            "confidence": 0.0,
            "raw_class": None,
            "source": "fallback_no_probs"
        }

    top_idx = int(probs.top1)
    confidence = float(probs.top1conf)
    raw_class = model.names[top_idx].lower()

    mapped = FRUIT_CLASS_MAP.get(
        raw_class,
        "generic_fruit"
    )

    return {
        "product_key": mapped,
        "confidence": round(confidence, 4),
        "raw_class": raw_class,
        "source": "fallback_classifier"
    }