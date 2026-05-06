# Fresh QA Debug Artifacts

This build can save debug images for every fresh-product scan.

## Enable

### In Streamlit
Turn on:

```text
Save debug masks + overlay
```

### In code

```python
result = infer_fresh_product(
    model_key="banana_ripeness",
    image=image,
    product_key="banana",
    save_debug=True,
)
```

### Environment variable

```bash
SAVE_DEBUG_ARTIFACTS=true
```

## Output folder

```text
data/debug_artifacts/<product_key>/<timestamp>_<result>_<scan_id>/
```

## Files saved

```text
original.jpg              original downsampled input
corrected.jpg             after color-reference correction
crop.jpg                  YOLO crop used for QA
 yellow_mask.png          yellow/ripeness mask
 green_mask.png           green/unripe mask
 brown_black_mask.png     brown/black defect mask
 sticker_mask.png         sticker/logo excluded area
 dark_cluster_mask.png    connected dark clusters used for reject logic
overlay.jpg               visual demo overlay
```

## Overlay legend

```text
yellow  = yellow/ripeness region
green   = green/unripe region
red     = brown/black defect region
magenta = large dark clusters
blue    = sticker/logo excluded area
```
