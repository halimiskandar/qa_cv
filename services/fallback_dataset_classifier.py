
"""
Fallback dataset-based classifier.

Intended flow:
1. YOLO tries to identify fruit.
2. If confidence low or unknown:
   - use embedding classifier
   - compare against Fruits-360 / Kaggle reference embeddings
3. Route to closest fruit family profile.

Current implementation:
- placeholder scaffold
"""

def resolve_unknown_fruit(image):
    return {
        "fallback_used": True,
        "resolved_family": "generic_fruit",
        "source": "dataset_embedding_fallback"
    }
