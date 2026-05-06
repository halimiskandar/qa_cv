# Banana Streamlit Drive Label Collector

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` from `.env.example`.

3. Put your downloaded Google service account JSON in the project root and name it:

```text
service_account.json
```

4. Share your Google Drive dataset folder with the service account email as **Editor**.

5. Run:

```bash
streamlit run app_streamlit.py
```

## Important

`GOOGLE_DRIVE_FOLDER_ID` must be the folder ID from a URL like:

```text
https://drive.google.com/drive/folders/FOLDER_ID_HERE
```

Do not paste a Google Doc, Sheet, or file ID. It must be a Drive folder.

## Auto-detect Fresh QA update

This package includes the new auto-detect product routing layer. Read:

```text
AUTO_DETECT_FRESH_QA.md
```

Key new files:

```text
services/fresh_auto_detect.py
services/fresh_api.py
services/fresh_config.py
```

The app now supports:

```text
✨ Auto Detect
```

Auto routing priority:

```text
explicit product_key
→ optional product_name keyword
→ YOLO detected class
→ generic_fresh fallback
```

Note: model weights and credential files are intentionally not included in this package. Put your `models/yolov8n.pt`, `.env`, and Google service account files back into your local repo as needed.
