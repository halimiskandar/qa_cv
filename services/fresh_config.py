"""Reusable Fresh QA configuration.

Design:
- product-specific profiles for known items like banana/apple/tomato/leafy veg
- generic fallback profiles for unseen fresh items detected by YOLO or entered by name
- thresholds live here so the inference logic stays reusable
"""

FRESH_PRODUCT_CONFIG = {
    "banana": {
        "display_name": "Banana",
        "detector_classes": ["banana"],
        "use_reference_card": True,
        "remove_sticker": True,
        "decision_profile": "banana",
        "generic_family": "fruit",
        "plastic": {
            "yellow_hsv": ((18, 45, 50), (38, 255, 255)),
            "green_hsv_1": ((35, 35, 35), (90, 255, 255)),
            "green_hsv_2": ((25, 15, 40), (50, 220, 245)),
            "unripe_green_threshold": 0.08,
            "ready_green_max": 0.05,
            "minor_brown_max": 0.12,
            "reject_brown_threshold": 0.20,
            "max_dark_cluster_ratio": 0.015,
            "max_dark_cluster_count": 5,
            "cluster_reject_brown_ratio": 0.055,
        },
        "no_plastic": {
            "yellow_hsv": ((18, 50, 50), (38, 255, 255)),
            "green_hsv_1": ((35, 40, 40), (85, 255, 255)),
            "green_hsv_2": None,
            "unripe_green_threshold": 0.25,
            "ready_green_max": 0.20,
            "minor_brown_max": 0.18,
            "reject_brown_threshold": 0.24,
            "max_dark_cluster_ratio": 0.020,
            "max_dark_cluster_count": 5,
            "cluster_reject_brown_ratio": 0.055,
        },
    },
    "apple": {
        "display_name": "Apple",
        "detector_classes": ["apple"],
        "use_reference_card": True,
        "remove_sticker": True,
        "decision_profile": "generic_bruise",
        "generic_family": "fruit",
        "plastic": {
            "reject_brown_threshold": 0.10,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.035,
        },
        "no_plastic": {
            "reject_brown_threshold": 0.12,
            "max_dark_cluster_ratio": 0.018,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.040,
        },
    },
    "tomato": {
        "display_name": "Tomato",
        "detector_classes": ["tomato"],
        "use_reference_card": True,
        "remove_sticker": False,
        "decision_profile": "generic_bruise",
        "generic_family": "soft_fruit",
        "plastic": {
            "reject_brown_threshold": 0.08,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 3,
            "cluster_reject_brown_ratio": 0.030,
        },
        "no_plastic": {
            "reject_brown_threshold": 0.10,
            "max_dark_cluster_ratio": 0.015,
            "max_dark_cluster_count": 3,
            "cluster_reject_brown_ratio": 0.035,
        },
    },
    "leafy_veg": {
        "display_name": "Leafy Vegetable",
        "detector_classes": ["broccoli", "lettuce", "cabbage", "spinach", "leafy_veg"],
        "use_reference_card": True,
        "remove_sticker": False,
        "decision_profile": "leafy_veg",
        "generic_family": "leafy_veg",
        "plastic": {
            "yellowing_max": 0.10,
            "reject_brown_threshold": 0.05,
            "max_dark_cluster_ratio": 0.010,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.025,
        },
        "no_plastic": {
            "yellowing_max": 0.14,
            "reject_brown_threshold": 0.07,
            "max_dark_cluster_ratio": 0.014,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.030,
        },
    },
    "generic_fruit": {
        "display_name": "Generic Fruit",
        "detector_classes": ["apple", "orange", "banana", "pear"],
        "use_reference_card": True,
        "remove_sticker": True,
        "decision_profile": "generic_bruise",
        "generic_family": "fruit",
        "plastic": {
            "reject_brown_threshold": 0.10,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.035,
        },
        "no_plastic": {
            "reject_brown_threshold": 0.12,
            "max_dark_cluster_ratio": 0.018,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.040,
        },
    },
    "generic_soft_fruit": {
        "display_name": "Generic Soft Fruit",
        "detector_classes": ["tomato"],
        "use_reference_card": True,
        "remove_sticker": False,
        "decision_profile": "generic_bruise",
        "generic_family": "soft_fruit",
        "plastic": {
            "reject_brown_threshold": 0.08,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 3,
            "cluster_reject_brown_ratio": 0.030,
        },
        "no_plastic": {
            "reject_brown_threshold": 0.10,
            "max_dark_cluster_ratio": 0.015,
            "max_dark_cluster_count": 3,
            "cluster_reject_brown_ratio": 0.035,
        },
    },
    "generic_leafy_veg": {
        "display_name": "Generic Leafy Vegetable",
        "detector_classes": ["broccoli", "lettuce", "cabbage", "spinach"],
        "use_reference_card": True,
        "remove_sticker": False,
        "decision_profile": "leafy_veg",
        "generic_family": "leafy_veg",
        "plastic": {
            "yellowing_max": 0.12,
            "reject_brown_threshold": 0.06,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.028,
        },
        "no_plastic": {
            "yellowing_max": 0.16,
            "reject_brown_threshold": 0.08,
            "max_dark_cluster_ratio": 0.016,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.034,
        },
    },
    "generic_fresh": {
        "display_name": "Generic Fresh Item",
        "detector_classes": [],
        "use_reference_card": True,
        "remove_sticker": False,
        "decision_profile": "generic_bruise",
        "generic_family": "unknown_fresh",
        "plastic": {
            "reject_brown_threshold": 0.09,
            "max_dark_cluster_ratio": 0.012,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.032,
        },
        "no_plastic": {
            "reject_brown_threshold": 0.12,
            "max_dark_cluster_ratio": 0.018,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.040,
        },
    },
}

# COCO / common detector classes mapped into fresh QA profiles.
# Extend this when you train your own fresh detector.
DETECTOR_CLASS_TO_PRODUCT_KEY = {
    "banana": "banana",
    "apple": "apple",
    "orange": "generic_fruit",
    "broccoli": "leafy_veg",
    "carrot": "generic_fresh",
    "tomato": "tomato",
    "lettuce": "leafy_veg",
    "cabbage": "leafy_veg",
    "spinach": "leafy_veg",
}

# Optional product-name fallback when detector class is unavailable.
PRODUCT_NAME_KEYWORDS = {
    "banana": ["banana", "pisang"],
    "apple": ["apple", "apel"],
    "tomato": ["tomato", "tomat"],
    "leafy_veg": ["lettuce", "selada", "spinach", "bayam", "kangkung", "sawi", "cabbage", "kol", "broccoli", "brokoli"],
    "generic_fruit": ["orange", "jeruk", "pear", "pir", "mango", "mangga", "dragon fruit", "buah naga"],
    "generic_fresh": ["fresh", "buah", "sayur", "vegetable", "fruit"],
}
