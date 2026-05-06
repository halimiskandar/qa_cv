"""Small API-facing wrapper for Fresh QA.

Use this from FastAPI/Flask/Streamlit so the outside world has one function.
"""

import base64
import io
from typing import Optional

from PIL import Image

from services.fresh_auto_detect import infer_auto_fresh_product
from services.fresh_inference import infer_fresh_product


def image_from_base64(image_base64: str) -> Image.Image:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    raw = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def run_fresh_qa_from_image(
    image: Image.Image,
    product_key: Optional[str] = None,
    product_name: Optional[str] = None,
    auto_detect: bool = True,
    model_key: str = "banana_ripeness",
    save_debug: bool = False,
) -> dict:
    """Run fresh QA with explicit product_key or auto-detect fallback."""
    if auto_detect or not product_key:
        return infer_auto_fresh_product(
            image=image,
            model_key=model_key,
            product_key=product_key,
            product_name=product_name,
            save_debug=save_debug,
            allow_auto_detect=auto_detect,
        )

    return infer_fresh_product(
        model_key=model_key,
        image=image,
        product_key=product_key,
        save_debug=save_debug,
    )


def run_fresh_qa_from_payload(payload: dict) -> dict:
    """Expected payload:

    {
      "image_base64": "...",
      "product_key": "banana",          # optional
      "product_name": "Pisang Cavendish",# optional
      "auto_detect": true,
      "save_debug": false
    }
    """
    image_base64 = payload.get("image_base64")
    if not image_base64:
        raise ValueError("payload.image_base64 is required")

    image = image_from_base64(image_base64)
    return run_fresh_qa_from_image(
        image=image,
        product_key=payload.get("product_key"),
        product_name=payload.get("product_name"),
        auto_detect=payload.get("auto_detect", True),
        model_key=payload.get("model_key", "banana_ripeness"),
        save_debug=payload.get("save_debug", False),
    )
