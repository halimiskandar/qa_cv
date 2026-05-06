import json
import os
from pathlib import Path
from typing import Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_value(key: str, default=None):
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


def _candidate_paths(path_text: str):
    raw = Path(path_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            PROJECT_ROOT / raw,
            PROJECT_ROOT.parent / raw,
        ])
    # preserve order, remove duplicates
    seen = set()
    unique = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _load_service_account_info() -> Tuple[Optional[dict], Optional[str]]:
    raw_json = _env_value("GOOGLE_SERVICE_ACCOUNT_JSON")
    json_path = _env_value("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

    if raw_json:
        try:
            return json.loads(raw_json), None
        except Exception as exc:
            return None, f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {exc}"

    if json_path:
        checked = []
        for p in _candidate_paths(json_path):
            checked.append(str(p))
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f), None
                except Exception as exc:
                    return None, f"Could not read service account file at {p}: {exc}"
        return None, "Service account file not found. Checked: " + " | ".join(checked)

    return None, "Missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON."


def get_drive_service():
    client_file = os.getenv("GOOGLE_OAUTH_CLIENT_FILE", "credentials.json")
    token_file = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def verify_folder_access(service, folder_id: str) -> dict:
    return service.files().get(
        fileId=folder_id,
        fields="id, name, mimeType, webViewLink",
        supportsAllDrives=True,
    ).execute()


def find_or_create_folder(service, folder_name: str, parent_folder_id: str) -> str:
    safe_name = _escape_drive_query_value(folder_name)
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{safe_name}' and trashed=false and '{parent_folder_id}' in parents"
    )

    resp = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def upload_file_to_drive(local_path: str, filename: str, folder_id: str, mime_type: str = "image/jpeg") -> str:
    service = get_drive_service()
    metadata = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=False)
    created = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, webViewLink, webContentLink",
        supportsAllDrives=True,
    ).execute()
    return created.get("webViewLink") or created.get("webContentLink") or created["id"]


def upload_image_to_label_folder(local_path: str, filename: str, label: str, root_folder_id: Optional[str] = None) -> str:
    root_folder_id = root_folder_id or _env_value("GOOGLE_DRIVE_FOLDER_ID") or _env_value("DRIVE_PARENT_FOLDER_ID")
    if not root_folder_id:
        raise RuntimeError("Missing GOOGLE_DRIVE_FOLDER_ID in .env")

    service = get_drive_service()
    folder_meta = verify_folder_access(service, root_folder_id)
    if folder_meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID exists but is not a Google Drive folder.")

    label_folder_id = find_or_create_folder(service, label, root_folder_id)
    return upload_file_to_drive(local_path, filename, folder_id=label_folder_id, mime_type="image/jpeg")


def get_drive_debug_status() -> dict:
    status = {
        "GOOGLE_DRIVE_FOLDER_ID_set": bool(_env_value("GOOGLE_DRIVE_FOLDER_ID") or _env_value("DRIVE_PARENT_FOLDER_ID")),
        "GOOGLE_SERVICE_ACCOUNT_FILE": _env_value("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json"),
        "service_account_file_found": False,
        "service_account_email": None,
        "folder_access_ok": False,
        "folder_name": None,
        "error": None,
    }
    info, err = _load_service_account_info()
    if err:
        status["error"] = err
        return status
    status["service_account_file_found"] = True
    status["service_account_email"] = info.get("client_email") if info else None

    folder_id = _env_value("GOOGLE_DRIVE_FOLDER_ID") or _env_value("DRIVE_PARENT_FOLDER_ID")
    if not folder_id:
        status["error"] = "Missing GOOGLE_DRIVE_FOLDER_ID in .env"
        return status
    try:
        service = get_drive_service()
        folder = verify_folder_access(service, folder_id)
        status["folder_access_ok"] = True
        status["folder_name"] = folder.get("name")
    except Exception as exc:
        status["error"] = str(exc)
    return status
