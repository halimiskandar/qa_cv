import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

from services.google_drive import get_drive_debug_status, upload_image_to_label_folder
from services.inference import run_inference
from services.fresh_inference import infer_fresh_product
from services.fresh_auto_detect import infer_auto_fresh_product
from services.fresh_config import FRESH_PRODUCT_CONFIG
from utils.image import resize_keep_aspect

load_dotenv()

APP_TITLE = "Banana QA Label Collector"
MODEL_TYPE = "banana_ripeness"
DOWNSAMPLED_LONGEST_SIDE = int(os.getenv("DOWNSAMPLED_LONGEST_SIDE", "640"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
LOCAL_IMAGE_DIR = Path(os.getenv("LOCAL_IMAGE_DIR", "data/review_images"))

TRAINING_LABELS = [
    "ready_to_send",
    "bruised",
    "stalk_issue",
    "overripe_black_spots",
    "unripe_too_green",
    "too_much_glare",
    "photo_blurry",
    "plastic_reflection",
    "wrong_item_no_banana",
    "cut_off",
    "too_small_or_far",
    "manual_review_needed",
    "retake_photo",
    "other",
]

PREDICTION_TO_TRAINING_LABEL = {
    "ready_to_send": "ready_to_send",
    "reject": "overripe_black_spots",
    "unripe": "unripe_too_green",
    "manual_review": "manual_review_needed",
    "retake_photo": "retake_photo",
    "no_banana_detected": "wrong_item_no_banana",
}

REJECT_REASON_TO_TRAINING_LABEL = {
    "too_much_glare": "too_much_glare",
    "photo_too_blurry": "photo_blurry",
    "banana_cut_off": "cut_off",
    "banana_too_small_or_far": "too_small_or_far",
    "too_green": "unripe_too_green",
    "too_many_dark_spots": "overripe_black_spots",
    "visible_dark_bruise_clusters": "overripe_black_spots",
    "large_dark_bruise_cluster": "overripe_black_spots",
    "no_banana_detected": "wrong_item_no_banana",
}

GENERIC_FRESH_LABELS = [
    "ready_to_send",
    "bruised",
    "overripe_black_spots",
    "yellowing",
    "mold_or_spoilage",
    "too_much_glare",
    "photo_blurry",
    "wrong_item",
    "too_small_or_far",
    "manual_review_needed",
    "retake_photo",
    "other",
]

GENERIC_FRESH_PREDICTION_MAP = {
    "ready_to_send": "ready_to_send",
    "reject": "bruised",
    "manual_review": "manual_review_needed",
    "retake_photo": "retake_photo",
}

GENERIC_FRESH_REJECT_MAP = {
    "visible_dark_bruise_clusters": "bruised",
    "too_many_dark_spots": "bruised",
    "leaf_yellowing": "yellowing",
    "too_much_glare": "too_much_glare",
    "photo_too_blurry": "photo_blurry",
}
PRODUCT_CONFIG = {
    "auto": {
        "model_type": "banana_ripeness",
        "product_key": None,
        "training_labels": GENERIC_FRESH_LABELS,
        "prediction_map": GENERIC_FRESH_PREDICTION_MAP,
        "reject_map": GENERIC_FRESH_REJECT_MAP,
        "use_fresh_pipeline": True,
        "auto_detect": True,
    },
    "banana": {
        "model_type": "banana_ripeness",
        "product_key": "banana",
        "training_labels": TRAINING_LABELS,
        "prediction_map": PREDICTION_TO_TRAINING_LABEL,
        "reject_map": REJECT_REASON_TO_TRAINING_LABEL,
        "use_fresh_pipeline": False,
    },
    "apple": {
        "model_type": "banana_ripeness",  # replace with your multi-item detector when ready
        "product_key": "apple",
        "training_labels": GENERIC_FRESH_LABELS,
        "prediction_map": GENERIC_FRESH_PREDICTION_MAP,
        "reject_map": GENERIC_FRESH_REJECT_MAP,
        "use_fresh_pipeline": True,
    },
    "tomato": {
        "model_type": "banana_ripeness",  # replace with your multi-item detector when ready
        "product_key": "tomato",
        "training_labels": GENERIC_FRESH_LABELS,
        "prediction_map": GENERIC_FRESH_PREDICTION_MAP,
        "reject_map": GENERIC_FRESH_REJECT_MAP,
        "use_fresh_pipeline": True,
    },
    "leafy_veg": {
        "model_type": "banana_ripeness",  # replace with your multi-item detector when ready
        "product_key": "leafy_veg",
        "training_labels": GENERIC_FRESH_LABELS,
        "prediction_map": GENERIC_FRESH_PREDICTION_MAP,
        "reject_map": GENERIC_FRESH_REJECT_MAP,
        "use_fresh_pipeline": True,
    },
    "meat": {
        "model_type": "meat_fat_ratio",
        "product_key": "meat",
        "training_labels": [
            "ready_to_send",
            "too_much_fat",
            "too_dark_spoiled",
            "damaged_bruised",
            "too_much_glare",
            "photo_blurry",
            "manual_review_needed",
            "retake_photo",
            "other",
        ],
        "prediction_map": {
            "good": "ready_to_send",
            "spoiled": "too_dark_spoiled",
            "fat": "too_much_fat",
            "manual_review": "manual_review_needed",
        },
        "reject_map": {
            "too_dark": "too_dark_spoiled",
            "too_much_fat": "too_much_fat",
            "too_much_glare": "too_much_glare",
        },
        "use_fresh_pipeline": False,
    }
}


st.set_page_config(page_title=APP_TITLE, page_icon="🍌", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 6rem; max-width: 720px;}
      .qa-card {border: 1px solid #eee; border-radius: 18px; padding: 14px; background: white; box-shadow: 0 1px 8px rgba(0,0,0,0.04); margin-bottom: 12px;}
      .big-status {font-size: 1.25rem; font-weight: 800; margin-bottom: 4px;}
      .small-muted {color: #666; font-size: 0.9rem;}
      .stButton > button {width: 100%; border-radius: 14px; min-height: 48px; font-weight: 700;}
      div[role="radiogroup"] label {padding: 8px 10px; border-radius: 12px;}
      @media (max-width: 640px) {
        .block-container {padding-left: 0.75rem; padding-right: 0.75rem;}
        .big-status {font-size: 1.15rem;}
        h1 {font-size: 1.65rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)




def init_state():
    for key, value in {
        "file_signature": None,
        "inference_id": None,
        "result": None,
        "image_path": None,
        "filename": None,
        "drive_url": None,
        "selected_label": None,
        "uploaded_to_drive": False,
        "last_error": None,
    }.items():
        st.session_state.setdefault(key, value)


def prepare_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def file_signature(uploaded_file) -> str:
    return f"{getattr(uploaded_file, 'name', 'camera')}_{getattr(uploaded_file, 'size', 0)}"


def save_downsampled_image(image: Image.Image, inference_id: str, label_hint: str = "unlabeled") -> tuple[str, str]:
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    downsampled = resize_keep_aspect(image, longest_side=DOWNSAMPLED_LONGEST_SIDE)
    filename = f"{label_hint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{inference_id}.jpg"
    local_path = LOCAL_IMAGE_DIR / filename
    downsampled.save(local_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return str(local_path), filename


def suggested_training_label(result: dict) -> str:
    reason = result.get("reject_reason")
    if reason in REJECT_REASON_TO_TRAINING_LABEL:
        return REJECT_REASON_TO_TRAINING_LABEL[reason]
    if reason in TRAINING_LABELS:
        return reason
    return PREDICTION_TO_TRAINING_LABEL.get(result.get("class"), "manual_review_needed")


def get_status(result: dict) -> tuple[str, str]:
    accepted = result.get("is_accepted")
    needs_manual_review = result.get("needs_manual_review", False)
    result_class = result.get("class")
    reason = result.get("reject_reason")

    if accepted is True:
        return "✅ ACCEPTED", "Suggested folder: `ready_to_send`"

    if needs_manual_review is True:
        return "👀 MANUAL REVIEW", f"Suggested issue: `{reason or 'manual_review_needed'}`"

    if accepted is False:
        return "❌ AUTO REJECT", f"Reason: `{reason or result_class}`"

    return "👀 MANUAL REVIEW", f"Suggested issue: `{reason or 'manual_review_needed'}`"


def show_result(result: dict):
    confidence = result.get("confidence")
    quality = result.get("quality_metrics") or {}
    color = result.get("color_metrics") or {}
    defects = result.get("defect_metrics") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Conf.", "-" if confidence is None else f"{confidence:.2f}")
    c2.metric("Blur", quality.get("blur_score", "-"))
    c3.metric("Glare", quality.get("glare_ratio", "-"))

    debug_artifacts = result.get("debug_artifacts") or {}
    overlay_path = (debug_artifacts.get("files") or {}).get("overlay")
    if overlay_path and Path(overlay_path).exists():
        st.image(overlay_path, caption="Debug overlay: yellow=ripe, green=unripe, red=defect, magenta=large clusters, blue=sticker", use_container_width=True)

    routing = result.get("routing") or {}
    profile = result.get("product_profile") or {}
    if routing or profile:
        st.caption(f"Route: `{routing.get('route_source', '-')}` → `{profile.get('product_key', result.get('product_key', '-'))}`")

    with st.expander("Prediction details"):
        debug_view = {
                "prediction": {
                    "class": result.get("class"),
                    "is_accepted": result.get("is_accepted"),
                    "needs_manual_review": result.get("needs_manual_review"),
                    "reject_reason": result.get("reject_reason"),
                    "instruction": result.get("instruction"),
                    "confidence": result.get("confidence"),
                },
                "routing": result.get("routing"),
                "product": {
                    "product_key": result.get("product_key"),
                    "detected_class": result.get("detected_class"),
                    "profile": result.get("product_profile"),
                },
                "crop": result.get("crop"),
                "quality_metrics": result.get("quality_metrics"),
                "color_metrics": result.get("color_metrics"),
                "defect_metrics": result.get("defect_metrics"),
                "thresholds": result.get("thresholds"),
                "packaging": result.get("packaging"),
                "debug_files": result.get("debug_artifacts", {}).get("files"),
            }

        st.json(debug_view)


def analyze_image(image: Image.Image):
    inference_id = str(uuid.uuid4())
    st.session_state.inference_id = inference_id
    st.session_state.drive_url = None
    st.session_state.selected_label = None
    st.session_state.uploaded_to_drive = False
    st.session_state.last_error = None

    started = time.time()
    downsampled = resize_keep_aspect(image, DOWNSAMPLED_LONGEST_SIDE)

    if config.get("auto_detect"):
        result = infer_auto_fresh_product(
            model_key=MODEL_TYPE,
            image=downsampled,
            product_name=st.session_state.get("product_name_hint") or None,
            save_debug=st.session_state.get("save_debug_artifacts", False),
            allow_auto_detect=True,
        )
    elif config.get("use_fresh_pipeline"):
        result = infer_fresh_product(
            model_key=MODEL_TYPE,
            image=downsampled,
            product_key=config["product_key"],
            save_debug=st.session_state.get("save_debug_artifacts", False),
        )
    else:
        result = run_inference(MODEL_TYPE, downsampled)

    label_hint = suggested_training_label(result)
    local_path, filename = save_downsampled_image(image, inference_id, label_hint=label_hint)
    result["latency_ms"] = round((time.time() - started) * 1000, 2)

    st.session_state.result = result
    st.session_state.image_path = local_path
    st.session_state.filename = filename


init_state()

PRODUCT_LABELS = {
    "auto": "✨ Auto Detect",
    "banana": "🍌 Banana",
    "apple": "🍎 Apple",
    "tomato": "🍅 Tomato",
    "leafy_veg": "🥬 Leafy Veg",
    "meat": "🥩 Meat",
}

product_type = st.selectbox(
    "Select Product",
    list(PRODUCT_CONFIG.keys()),
    format_func=lambda x: PRODUCT_LABELS.get(x, x),
)
if st.session_state.get("product_type") != product_type:
    st.session_state.product_type = product_type
    st.session_state.file_signature = None
    st.session_state.result = None
    st.session_state.image_path = None
    st.session_state.filename = None
    st.session_state.uploaded_to_drive = False

config = PRODUCT_CONFIG[product_type]

MODEL_TYPE = config["model_type"]
TRAINING_LABELS = config["training_labels"]
PREDICTION_TO_TRAINING_LABEL = config["prediction_map"]
REJECT_REASON_TO_TRAINING_LABEL = config["reject_map"]

st.title(f"{PRODUCT_LABELS.get(product_type, product_type)} QA Label Collector")
st.caption(f"Current mode: {product_type.upper()}")

if config.get("auto_detect"):
    st.session_state.product_name_hint = st.text_input(
        "Optional product name hint",
        value=st.session_state.get("product_name_hint", ""),
        placeholder="Example: pisang, apel fuji, tomat, selada",
        help="Optional. If filled, the app uses this before YOLO auto-detect. If empty, YOLO detects the item and falls back to generic fresh.",
    )
st.caption("Take or upload a photo. The app auto-predicts, then you confirm the correct folder for training data.")

st.session_state.save_debug_artifacts = st.toggle(
    "Save debug masks + overlay",
    value=st.session_state.get("save_debug_artifacts", True),
    help="Saves original/corrected/crop/mask/overlay images under data/debug_artifacts for demo and QA review.",
)

with st.expander("Google Drive check", expanded=False):
    if st.button("Check Drive setup"):
        st.json(get_drive_debug_status())
    st.caption("Your .env should contain GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json")

input_mode = st.segmented_control("Input", options=["Camera", "Upload"], default="Camera")
image_file = (
    st.camera_input(f"Take {product_type} photo")
    if input_mode == "Camera"
    else st.file_uploader(f"Upload {product_type} image", type=["jpg", "jpeg", "png", "webp"])
)
if image_file is not None:
    image = prepare_image(image_file)
    st.image(image, caption="Original photo", use_container_width=True)

    sig = file_signature(image_file)
    if sig != st.session_state.file_signature:
        st.session_state.file_signature = sig
        with st.spinner("Analyzing photo..."):
            analyze_image(image)
        st.rerun()

if st.session_state.result:
    result = st.session_state.result
    status, subtitle = get_status(result)
    default_label = suggested_training_label(result)
    default_idx = TRAINING_LABELS.index(default_label) if default_label in TRAINING_LABELS else 0

    st.markdown('<div class="qa-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-status">{status}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)
    if result.get("instruction"):
        st.info(result["instruction"])
    show_result(result)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("label_form", clear_on_submit=False):
        st.subheader("Confirm correct training label")
        prediction_correct = st.radio(
            "Is the prediction correct?",
            options=["true", "false"],
            horizontal=True,
            index=0 if result.get("is_accepted") is True else 1,
        )
        if prediction_correct == "true":
            selected_label = default_label
            st.success(f"Will save to Drive folder: `{selected_label}`")
        else:
            selected_label = st.selectbox("Choose correct Drive folder / training label", TRAINING_LABELS, index=default_idx)
            st.caption("The selected label decides the Google Drive subfolder.")
        submitted = st.form_submit_button("Save Downsampled Image to Drive")

    if submitted:
        try:
            with st.spinner(f"Uploading to Google Drive folder: {selected_label}..."):
                drive_url = upload_image_to_label_folder(st.session_state.image_path, st.session_state.filename, selected_label)
            st.session_state.drive_url = drive_url
            st.session_state.selected_label = selected_label
            st.session_state.uploaded_to_drive = True
            st.success(f"Saved to Google Drive label folder: `{selected_label}`")
            st.link_button("Open saved image", drive_url)
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error("Google Drive upload failed.")
            with st.expander("Show error detail"):
                st.code(str(exc))
            with open(st.session_state.image_path, "rb") as f:
                st.download_button("Download local downsampled image", f, file_name=Path(st.session_state.image_path).name, mime="image/jpeg")

    if st.session_state.uploaded_to_drive:
        st.info(f"Current saved label: `{st.session_state.selected_label}`")
        if st.session_state.drive_url:
            st.link_button("Open saved image", st.session_state.drive_url)
