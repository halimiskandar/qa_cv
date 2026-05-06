def decide_fresh_quality(product_key, product_cfg, quality, color, defect):
    plastic_key = "plastic" if quality["plastic_mode"] == "with_plastic" else "no_plastic"
    thresholds = product_cfg[plastic_key]
    profile = product_cfg.get("decision_profile", "generic_bruise")

    result_class = "manual_review"
    is_accepted = None
    reject_reason = None
    instruction = None
    quality_warning = None

    if quality["touches_edge"]:
        quality_warning = f"{product_key}_touches_edge_result_may_be_less_accurate"

    max_glare_ratio = thresholds.get("max_glare_ratio", 0.18)

    if quality["glare_ratio"] > max_glare_ratio:
        if profile == "banana" and quality.get("plastic_mode") == "with_plastic":
            quality_warning = "high_glare_from_plastic"
        else:
            return _result(
                "retake_photo",
                False,
                "too_much_glare",
                "Please retake photo with less reflection.",
                quality_warning,
                thresholds
            )
    surface_cfg = product_cfg.get("surface", {})

    blur_threshold = 40

    if surface_cfg.get("texture_detector", False):
        blur_threshold = 18

    if quality["blur_score"] < blur_threshold:
        return _result("retake_photo", False, "photo_too_blurry", "Please retake photo. Hold camera steady.", quality_warning, thresholds)
    min_coverage_ratio = thresholds.get("min_coverage_ratio", 0.25)

    if quality["coverage_ratio"] < min_coverage_ratio:
        return _result(
            "retake_photo",
            False,
            f"{product_key}_too_small",
            "Please retake photo closer to the item.",
            quality_warning,
            thresholds
        )
    
    brown = color.get("brown_black_ratio", 0)
    green = color.get("green_ratio", 0)
    yellow = color.get("yellow_ratio", 0)
    largest_cluster = defect.get("largest_dark_cluster_ratio", 0)
    cluster_count = defect.get("dark_cluster_count", 0)

    max_cluster = thresholds.get("max_dark_cluster_ratio", 0.02)
    max_cluster_count = thresholds.get("max_dark_cluster_count", 5)
    cluster_brown_ratio = thresholds.get("cluster_reject_brown_ratio", 0.055)

    if profile == "banana":
        green_signal = max(
            color.get("green_ratio", 0),
            color.get("green_ratio_normal", 0),
            color.get("green_ratio_plastic", 0),
        )

        yellow = color.get("yellow_ratio", 0)

        if yellow >= 0.25 and green_signal >= 0.025:
            return _result(
                "unripe",
                False,
                "too_green",
                None,
                quality_warning,
                thresholds
            )
        if green_signal > thresholds["unripe_green_threshold"]:
            return _result("unripe", False, "too_green", None, quality_warning, thresholds)

        if largest_cluster > max_cluster or (brown > cluster_brown_ratio and cluster_count >= max_cluster_count):
            return _result("reject", False, "visible_dark_bruise_clusters", None, quality_warning, thresholds)

        if brown > thresholds.get("reject_brown_threshold", 0.15):
            return _result("reject", False, "too_many_dark_spots", None, quality_warning, thresholds)

        if yellow >= 0.45 and green_signal <= thresholds["ready_green_max"] and brown <= thresholds["minor_brown_max"]:
            return _result("ready_to_send", True, None, None, quality_warning, thresholds)

        if yellow >= 0.35 and green_signal > 0.04:
            return _result("almost_ripe", None, "mixed_ripeness", None, quality_warning, thresholds)
        
    elif profile == "leafy_veg":
        yellow_leaf = color.get("yellow_leaf_ratio", 0)
        brown_leaf = color.get("brown_leaf_ratio", 0)
        black_leaf = color.get("black_leaf_ratio", 0)

        if (
            yellow_leaf > thresholds.get("yellowing_max", 0.12)
            or brown_leaf > thresholds.get("brown_leaf_max", 0.08)
            or black_leaf > thresholds.get("black_leaf_max", 0.04)
        ):
            return _result("reject", False, "leaf_spoilage_detected", None, quality_warning, thresholds)

        return _result("ready_to_send", True, None, None, quality_warning, thresholds)

    else:  # generic bruise model for apple/tomato/etc.
        return _result("ready_to_send", True, None, None, quality_warning, thresholds)

    return _result("manual_review", None, "uncertain_color_quality", "Please review manually. Color quality is uncertain.", quality_warning, thresholds)


def _result(result_class, is_accepted, reason, instruction, quality_warning, thresholds):
    return {
        "class": result_class,
        "is_accepted": is_accepted,
        "needs_manual_review": result_class in ["manual_review", "retake_photo"],
        "reject_reason": reason,
        "instruction": instruction,
        "quality_warning": quality_warning,
        "thresholds": thresholds,
    }
