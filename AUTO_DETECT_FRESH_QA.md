# Auto-detect Fresh QA

This version supports scalable product routing:

```text
image
→ optional product_name/product_key
→ YOLO auto-detect fresh item
→ product-specific profile if known
→ generic fallback profile if unknown
→ shared defect feature engine
→ accept / reject / manual review
```

## Main entrypoint

Use this from app/API:

```python
from services.fresh_api import run_fresh_qa_from_image

result = run_fresh_qa_from_image(
    image=image,
    product_key=None,              # optional explicit override
    product_name="apel fuji",      # optional hint
    auto_detect=True,
    model_key="banana_ripeness",   # currently points to YOLOv8n in active_model.json
    save_debug=True,
)
```

## Routing priority

1. `product_key`, when explicitly provided
2. `product_name` keyword mapping, for example `pisang`, `apel`, `tomat`, `selada`
3. YOLO detected class, for example `banana`, `apple`, `orange`, `broccoli`
4. `generic_fresh` fallback

## Config files

Edit this file to add more products and thresholds:

```text
services/fresh_config.py
```

Important objects:

```python
FRESH_PRODUCT_CONFIG
DETECTOR_CLASS_TO_PRODUCT_KEY
PRODUCT_NAME_KEYWORDS
```

## Current product profiles

```text
banana              specific banana ripeness/bruising profile
apple               apple bruise profile
tomato              tomato / soft fruit bruise profile
leafy_veg           leafy vegetable yellowing + dark spot profile
generic_fruit       fallback for fruits like orange/pear/mango
generic_soft_fruit  fallback for tomato-like soft fruit
generic_leafy_veg   fallback for green vegetables
generic_fresh       final fallback if item is unknown
meat                still uses existing meat_fat_ratio path
```

## Streamlit

The product selector now has:

```text
✨ Auto Detect
```

In this mode, you may leave product name empty. The app will use YOLO and fallback to generic fresh if no known item is detected.

## Important limitation

Your current `models/yolov8n.pt` is a COCO detector. It can detect some items like banana, apple, orange, broccoli and carrot, but it will not reliably detect all fresh SKUs. For the competition demo, this is fine as an auto-routing MVP. Later, train a fresh detector with classes like:

```text
banana
apple
tomato
leafy_veg
citrus
root_veg
meat
unknown_fresh
```

The shared QA features still run even for unknown items through `generic_fresh`.
