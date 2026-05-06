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
            "yellow_hsv": ((18, 35, 50), (42, 255, 255)),
            "green_hsv_1": ((32, 35, 35), (95, 255, 255)),
            "green_hsv_2": ((25, 18, 35), (60, 220, 255)),
            "unripe_green_threshold": 0.04,
            "ready_green_max": 0.05,
            "minor_brown_max": 0.12,
            "reject_brown_threshold": 0.20,
            "max_dark_cluster_ratio": 0.025,
            "max_dark_cluster_count": 5,
            "cluster_reject_brown_ratio": 0.07,
            "min_coverage_ratio": 0.15,
            "max_glare_ratio": 0.60,
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
            "brown_leaf_max": 0.08,
            "black_leaf_max": 0.04,
        },
        "surface": {
        "glare_sensitive": False,
        "porous": False,
        "speckled": False,
        "texture_detector": False,
        "morph_kernel": 5,
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
        "surface": {
            "glare_sensitive": True,
            "porous": False,
            "speckled": False,
            "texture_detector": False,
            "morph_kernel": 5,
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
        "surface": {
            "glare_sensitive": True,
            "porous": False,
            "speckled": False,
            "texture_detector": False,
            "morph_kernel": 5,
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
            "brown_leaf_max": 0.08,
            "black_leaf_max": 0.04,
        },
        "no_plastic": {
            "yellowing_max": 0.14,
            "reject_brown_threshold": 0.07,
            "max_dark_cluster_ratio": 0.014,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.030,
            "brown_leaf_max": 0.08,
            "black_leaf_max": 0.04,
        },
        "surface": {
            "glare_sensitive": False,
            "porous": False,
            "speckled": False,
            "texture_detector": False,
            "morph_kernel": 3,
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
        "surface": {
            "glare_sensitive": False,
            "porous": True,
            "speckled": True,
            "texture_detector": True,
            "morph_kernel": 3,
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
            "brown_leaf_max": 0.08,
            "black_leaf_max": 0.04,
        },
        "no_plastic": {
            "yellowing_max": 0.16,
            "reject_brown_threshold": 0.08,
            "max_dark_cluster_ratio": 0.016,
            "max_dark_cluster_count": 4,
            "cluster_reject_brown_ratio": 0.034,
            "brown_leaf_max": 0.08,
            "black_leaf_max": 0.04,
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
        "surface": {
            "glare_sensitive": True,
            "porous": False,
            "speckled": False,
            "texture_detector": False,
            "morph_kernel": 5,
        },
    },
    "asian_pear": {
    "display_name": "Asian Pear",
    "detector_classes": ["pear"],

    "use_reference_card": True,
    "remove_sticker": True,

    "decision_profile": "generic_bruise",
    "generic_family": "fruit",

    "plastic": {
        "reject_brown_threshold": 0.12,
        "max_dark_cluster_ratio": 0.035,
        "max_dark_cluster_count": 8,
        "cluster_reject_brown_ratio": 0.050,
    },

    "no_plastic": {
        "reject_brown_threshold": 0.14,
        "max_dark_cluster_ratio": 0.045,
        "max_dark_cluster_count": 10,
        "cluster_reject_brown_ratio": 0.055,
    },

    "surface": {
        "glare_sensitive": False,
        "porous": True,
        "speckled": True,
        "texture_detector": True,
        "morph_kernel": 3,
        },
    },"orange": {
    "display_name": "Orange",
    "detector_classes": ["orange", "fruit"],
    "decision_profile": "generic_bruise",
    "generic_family": "fruit",

    "use_reference_card": False,
    "remove_sticker": True,

    "plastic": {
        "reject_brown_threshold": 0.28,
        "max_dark_cluster_ratio": 0.09,
        "max_dark_cluster_count": 12,
        "cluster_reject_brown_ratio": 0.18,
        "max_glare_ratio": 0.35,
        "min_coverage_ratio": 0.18,
    },

    "no_plastic": {
        "reject_brown_threshold": 0.30,
        "max_dark_cluster_ratio": 0.10,
        "max_dark_cluster_count": 12,
        "cluster_reject_brown_ratio": 0.20,
        "max_glare_ratio": 0.40,
        "min_coverage_ratio": 0.18,
    },

    "surface": {
        "glare_sensitive": False,
        "porous": True,
        "speckled": True,
        "texture_detector": True,
        "morph_kernel": 5
    }
    }
}

# COCO / common detector classes mapped into fresh QA profiles.
# Extend this when you train your own fresh detector.
DETECTOR_CLASS_TO_PRODUCT_KEY = {
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
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
    "asian_pear": ["asian pear","pear","pir","singo pear","korean pear"],
    "leafy_veg": ["lettuce", "selada", "spinach", "bayam", "kangkung", "sawi", "cabbage", "kol", "broccoli", "brokoli"],
    "orange": ["orange", "jeruk"],
    "generic_fruit": [ "jeruk", "pear", "pir", "mango", "mangga", "dragon fruit", "buah naga"],
    "generic_fresh": ["fresh", "buah", "sayur", "vegetable", "fruit"],
}
