from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import httpx


API_BASE = "https://civitai.com/api/v1"
DOWNLOAD_HOSTS = {"civitai.com", "www.civitai.com"}
SUPPORTED_TYPES = {"Checkpoint", "LORA"}
SAFE_EXTENSIONS = {".safetensors"}
BLOCKED_SCAN_RESULTS = {"danger", "error"}
DEFAULT_MAX_DOWNLOAD_GB = 12.0


class CivitaiError(RuntimeError):
    """Raised when Civitai search or download cannot be completed safely."""


def _headers(api_token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Renewed-Fooocus-Civitai-Manager/1.0",
    }
    token = (api_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _format_is_safetensors(file_info: dict[str, Any]) -> bool:
    name = str(file_info.get("name") or "")
    metadata = file_info.get("metadata") or {}
    file_format = str(metadata.get("format") or file_info.get("format") or "").lower()
    return Path(name).suffix.lower() in SAFE_EXTENSIONS and "safetensor" in file_format


def _scan_is_acceptable(file_info: dict[str, Any]) -> bool:
    for key in ("virusScanResult", "pickleScanResult"):
        result = str(file_info.get(key) or "").strip().lower()
        if result in BLOCKED_SCAN_RESULTS:
            return False
    return True


def _choose_safe_file(version: dict[str, Any]) -> dict[str, Any] | None:
    files = list(version.get("files") or [])
    files.sort(key=lambda item: bool(item.get("primary")), reverse=True)
    for file_info in files:
        if not _format_is_safetensors(file_info):
            continue
        if not _scan_is_acceptable(file_info):
            continue
        if file_info.get("downloadUrl") or file_info.get("primary"):
            return file_info
    return None


def _is_fooocus_family(base_model: str) -> bool:
    normalized = (base_model or "").lower()
    return any(marker in normalized for marker in ("sdxl", "pony", "illustrious"))


def _iter_results(items: Iterable[dict[str, Any]], compatibility: str) -> Iterable[dict[str, Any]]:
    for model in items:
        if bool(model.get("nsfw")):
            continue
        model_type = str(model.get("type") or "")
        if model_type not in SUPPORTED_TYPES:
            continue

        for version in model.get("modelVersions") or []:
            base_model = str(version.get("baseModel") or "Unknown")
            compatible = _is_fooocus_family(base_model)
            if compatibility == "Fooocus-compatible (SDXL family)" and not compatible:
                continue

            safe_file = _choose_safe_file(version)
            if safe_file is None:
                continue

            version_id = int(version["id"])
            file_download_url = safe_file.get("downloadUrl")
            version_download_url = version.get("downloadUrl")
            download_url = file_download_url or version_download_url or f"https://civitai.com/api/download/models/{version_id}"

            yield {
                "model_id": int(model["id"]),
                "model_name": str(model.get("name") or f"Model {model['id']}"),
                "model_type": model_type,
                "version_id": version_id,
                "version_name": str(version.get("name") or f"Version {version_id}"),
                "base_model": base_model,
                "fooocus_compatible": compatible,
                "trained_words": list(version.get("trainedWords") or []),
                "download_url": str(download_url),
                "file_id": safe_file.get("id"),
                "file_name": str(safe_file.get("name") or f"civitai-{version_id}.safetensors"),
                "size_kb": float(safe_file.get("sizeKB") or safe_file.get("sizeKb") or 0),
                "hashes": dict(safe_file.get("hashes") or {}),
                "virus_scan": safe_file.get("virusScanResult"),
                "pickle_scan": safe_file.get("pickleScanResult"),
                "model_url": f"https://civitai.com/models/{model['id']}?modelVersionId={version_id}",
            }


def search_models(
    query: str,
    model_type: str,
    sort: str = "Most Downloaded",
    period: str = "AllTime",
    compatibility: str = "Fooocus-compatible (SDXL family)",
    limit: int = 25,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    if model_type not in SUPPORTED_TYPES:
        raise CivitaiError(f"Unsupported model type: {model_type}")
    if not 1 <= limit <= 100:
        raise CivitaiError("Search limit must be between 1 and 100.")

    params = {
        "limit": limit,
        "types": model_type,
        "sort": sort,
        "period": period,
        "query": (query or "").strip(),
        "nsfw": "false",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
            response = client.get(f"{API_BASE}/models", params=params, headers=_headers(api_token))
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CivitaiError(f"Civitai search failed: {exc}") from exc

    results = list(_iter_results(payload.get("items") or [], compatibility))
    return results[:100]


def extract_version_id(reference: str | int) -> int:
    """Extract a Civitai model-version ID from a number or supported Civitai URL."""
    value = str(reference).strip()
    if value.isdigit():
        return int(value)

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or hostname not in DOWNLOAD_HOSTS:
        raise CivitaiError("Use a Civitai model-version URL or numeric version ID.")

    query = parse_qs(parsed.query)
    for key, values in query.items():
        if key.lower() == "modelversionid" and values and str(values[0]).isdigit():
            return int(values[0])

    for pattern in (r"/api/download/models/(\d+)", r"/api/v1/model-versions/(\d+)"):
        match = re.search(pattern, parsed.path, re.IGNORECASE)
        if match:
            return int(match.group(1))

    raise CivitaiError(
        "The Civitai reference does not include a modelVersionId. "
        "Open the desired version and copy its download URL or version ID."
    )


def get_model_version(reference: str | int, api_token: str | None = None) -> dict[str, Any]:
    """Resolve one Civitai model version into a downloadable manager record."""
    version_id = extract_version_id(reference)
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
            response = client.get(f"{API_BASE}/model-versions/{version_id}", headers=_headers(api_token))
            response.raise_for_status()
            version = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CivitaiError(f"Could not resolve Civitai version {version_id}: {exc}") from exc

    model = version.get("model") or {}
    model_type = str(model.get("type") or "")
    if model_type not in SUPPORTED_TYPES:
        raise CivitaiError(f"Civitai version {version_id} is not a checkpoint or LoRA.")
    if bool(model.get("nsfw")):
        raise CivitaiError("This manager does not install models marked NSFW.")

    safe_file = _choose_safe_file(version)
    if safe_file is None:
        raise CivitaiError("No acceptable .safetensors file was found for this model version.")

    model_id = int(version.get("modelId") or model.get("id") or 0)
    base_model = str(version.get("baseModel") or "Unknown")
    download_url = (
        safe_file.get("downloadUrl")
        or version.get("downloadUrl")
        or f"https://civitai.com/api/download/models/{version_id}"
    )

    return {
        "model_id": model_id,
        "model_name": str(model.get("name") or f"Model {model_id or version_id}"),
        "model_type": model_type,
        "version_id": version_id,
        "version_name": str(version.get("name") or f"Version {version_id}"),
        "base_model": base_model,
        "fooocus_compatible": _is_fooocus_family(base_model),
        "trained_words": list(version.get("trainedWords") or []),
        "download_url": str(download_url),
        "file_id": safe_file.get("id"),
        "file_name": str(safe_file.get("name") or f"civitai-{version_id}.safetensors"),
        "size_kb": float(safe_file.get("sizeKB") or safe_file.get("sizeKb") or 0),
        "hashes": dict(safe_file.get("hashes") or {}),
        "virus_scan": safe_file.get("virusScanResult"),
        "pickle_scan": safe_file.get("pickleScanResult"),
        "model_url": (
            f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
            if model_id
            else f"https://civitai.com/api/v1/model-versions/{version_id}"
        ),
    }


def _safe_filename(filename: str) -> str:
    filename = os.path.basename(filename.strip().replace("\\", "/"))
    filename = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", filename).strip(" .")
    if not filename:
        raise CivitaiError("Civitai did not provide a usable filename.")
    if Path(filename).suffix.lower() not in SAFE_EXTENSIONS:
        raise CivitaiError("Only .safetensors downloads are accepted.")
    return filename


def _filename_from_disposition(header: str | None, fallback: str) -> str:
    if not header:
        return _safe_filename(fallback)
    match = re.search(r"filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)", header, re.IGNORECASE)
    selected = (match.group(1) or match.group(2)) if match else fallback
    return _safe_filename(unquote(selected))


def _validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in DOWNLOAD_HOSTS:
        raise CivitaiError("Refused a download URL that is not hosted by Civitai.")


def _expected_sha256(record: dict[str, Any]) -> str | None:
    hashes = record.get("hashes") or {}
    value = hashes.get("SHA256") or hashes.get("sha256")
    return str(value).lower() if value else None


def download_model(
    record: dict[str, Any],
    destination_dir: str,
    api_token: str | None = None,
    overwrite: bool = False,
    max_download_gb: float | None = None,
) -> dict[str, Any]:
    if record.get("model_type") not in SUPPORTED_TYPES:
        raise CivitaiError("The selected item is not a checkpoint or LoRA.")
    if bool(record.get("nsfw")):
        raise CivitaiError("This manager does not install models marked NSFW.")

    url = str(record.get("download_url") or "")
    _validate_download_url(url)

    max_gb = max_download_gb
    if max_gb is None:
        max_gb = float(os.getenv("FOOOCUS_CIVITAI_MAX_GB", DEFAULT_MAX_DOWNLOAD_GB))
    if max_gb <= 0:
        raise CivitaiError("FOOOCUS_CIVITAI_MAX_GB must be greater than zero.")
    max_bytes = int(max_gb * 1024 ** 3)

    target_dir = Path(destination_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    fallback_name = _safe_filename(str(record.get("file_name") or ""))
    temp_path: Path | None = None

    try:
        with httpx.Client(timeout=httpx.Timeout(None, connect=30.0), follow_redirects=True) as client:
            with client.stream("GET", url, headers=_headers(api_token)) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length") or 0)
                if content_length and content_length > max_bytes:
                    raise CivitaiError(f"Download is larger than the configured {max_gb:g} GB limit.")

                filename = _filename_from_disposition(response.headers.get("Content-Disposition"), fallback_name)
                target_path = (target_dir / filename).resolve()
                if target_path.parent != target_dir:
                    raise CivitaiError("Refused an unsafe destination path.")
                if target_path.exists() and not overwrite:
                    raise CivitaiError(f"{filename} already exists. Enable overwrite to replace it.")

                digest = hashlib.sha256()
                downloaded = 0
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{filename}.", suffix=".part", dir=target_dir, delete=False
                ) as temporary:
                    temp_path = Path(temporary.name)
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise CivitaiError(f"Download exceeded the configured {max_gb:g} GB limit.")
                        digest.update(chunk)
                        temporary.write(chunk)

        actual_sha256 = digest.hexdigest().lower()
        expected_sha256 = _expected_sha256(record)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise CivitaiError("SHA-256 verification failed; the temporary file was removed.")

        os.replace(temp_path, target_path)
        temp_path = None

        metadata = {
            "source": "Civitai",
            "installed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "sha256": actual_sha256,
            **record,
        }
        metadata_path = target_path.with_suffix(target_path.suffix + ".civitai.json")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        return {
            "path": str(target_path),
            "metadata_path": str(metadata_path),
            "sha256": actual_sha256,
            "bytes": downloaded,
        }
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise CivitaiError("Civitai denied the download. Add an API token for gated or account-only assets.") from exc
        raise CivitaiError(f"Civitai download failed with HTTP {status}.") from exc
    except httpx.HTTPError as exc:
        raise CivitaiError(f"Civitai download failed: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def install_reference(
    reference: str | int,
    destination_dir: str,
    expected_type: str | None = None,
    api_token: str | None = None,
    overwrite: bool = False,
    max_download_gb: float | None = None,
) -> dict[str, Any]:
    """Resolve and install a Civitai version reference in one call."""
    record = get_model_version(reference, api_token=api_token)
    if expected_type and record["model_type"] != expected_type:
        raise CivitaiError(
            f"The selected Civitai item is {record['model_type']}, not the requested {expected_type}."
        )
    result = download_model(
        record=record,
        destination_dir=destination_dir,
        api_token=api_token,
        overwrite=overwrite,
        max_download_gb=max_download_gb,
    )
    return {"record": record, **result}


def _command_line() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Install Civitai assets for Renewed Fooocus.")
    parser.add_argument("--type", dest="model_type", choices=sorted(SUPPORTED_TYPES), required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-gb", type=float, default=None)
    args = parser.parse_args()

    token = os.getenv("CIVITAI_API_TOKEN", "").strip() or None
    failures = 0
    for reference in args.reference:
        print(f"[Civitai] Resolving {args.model_type}: {reference}", flush=True)
        try:
            result = install_reference(
                reference=reference,
                destination_dir=args.destination,
                expected_type=args.model_type,
                api_token=token,
                overwrite=args.overwrite,
                max_download_gb=args.max_gb,
            )
        except CivitaiError as exc:
            failures += 1
            print(f"[Civitai] Skipped: {exc}", flush=True)
            continue

        record = result["record"]
        size_gb = result["bytes"] / 1024**3
        print(
            f"[Civitai] Installed {record['model_name']} — {record['version_name']} "
            f"as {Path(result['path']).name} ({size_gb:.2f} GB)",
            flush=True,
        )
        words = ", ".join(record.get("trained_words") or [])
        if words:
            print(f"[Civitai] Trigger words: {words}", flush=True)

    if failures:
        print(f"[Civitai] {failures} requested asset(s) could not be installed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_command_line())
