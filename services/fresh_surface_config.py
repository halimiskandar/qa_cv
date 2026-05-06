
FRESH_SURFACE_CONFIG = {
    "banana": {
        "surface_type": "smooth_yellow",
        "glare_sensitive": False,
        "porous": False,
        "texture_detector": False,
        "morph_kernel": 5,
        "max_dark_cluster_ratio": 0.020,
        "max_dark_cluster_count": 5,
        "max_bruise_ratio": 0.10,
    },

    "orange": {
        "surface_type": "porous_citrus",
        "glare_sensitive": False,
        "porous": True,
        "texture_detector": True,
        "morph_kernel": 3,
        "max_dark_cluster_ratio": 0.045,
        "max_dark_cluster_count": 8,
        "max_bruise_ratio": 0.16,
    },

    "asian_pear": {
        "surface_type": "speckled_light_fruit",
        "glare_sensitive": False,
        "porous": True,
        "texture_detector": True,
        "morph_kernel": 3,
        "max_dark_cluster_ratio": 0.040,
        "max_dark_cluster_count": 10,
        "max_bruise_ratio": 0.14,
    },

    "generic_fruit": {
        "surface_type": "generic",
        "glare_sensitive": True,
        "porous": False,
        "texture_detector": False,
        "morph_kernel": 5,
        "max_dark_cluster_ratio": 0.025,
        "max_dark_cluster_count": 5,
        "max_bruise_ratio": 0.10,
    }
}
