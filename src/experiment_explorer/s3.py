"""S3 helpers for listing and downloading experiment folders."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import boto3
from botocore.exceptions import BotoCoreError, ClientError, LoginRefreshRequired

from experiment_explorer.config import get_aws_region
from experiment_explorer.paths import resolve_within

LOGGER = logging.getLogger(__name__)
AWS_LOGIN_MESSAGE = "AWS session expired. Reauthenticate with `aws login` and retry."
# Records the S3 ETag and size of each downloaded file, so changes are detected.
MANIFEST_NAME = ".s3_manifest.json"


class S3ExperimentError(RuntimeError):
    """Raised when S3 operations for experiments fail."""


class ExperimentNotFoundError(S3ExperimentError):
    """Raised when an experiment prefix has no files in the bucket."""

    def __init__(self, bucket: str, experiment_name: str, removed_count: int = 0) -> None:
        cleanup = (
            f"; removed {removed_count} cached file(s) that came from S3" if removed_count else ""
        )
        super().__init__(
            f"Experiment '{experiment_name}' was not found in bucket '{bucket}'{cleanup}. "
            "Use list_experiments to see valid names."
        )


@dataclass(frozen=True)
class S3ObjectInfo:
    """Metadata for a single S3 object."""

    key: str
    size_bytes: int
    last_modified: str
    etag: str = ""


def _create_s3_client() -> Any:
    """Create a boto3 S3 client using the default credential chain."""
    region = get_aws_region()
    try:
        if region:
            return boto3.client("s3", region_name=region)
        return boto3.client("s3")
    except Exception as exc:
        raise S3ExperimentError(
            "Failed to initialize AWS S3 client. "
            "Run `aws login` and ensure botocore[crt] is installed. "
            f"Details: {exc}"
        ) from exc


def _normalize_prefix(prefix: str) -> str:
    """Ensure prefix uses forward slashes and optional trailing slash."""
    cleaned = prefix.strip().lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        return f"{cleaned}/"
    return cleaned


def _format_client_error(exc: ClientError, *, bucket: str, prefix: str) -> str:
    error = exc.response.get("Error", {})
    code = error.get("Code", "Unknown")
    message = error.get("Message", str(exc))
    return (
        f"S3 operation failed for s3://{bucket}/{prefix} "
        f"({code}: {message}). "
        "Check AWS credentials with `aws login` or your configured profile."
    )


def _handle_s3_exception(exc: Exception, *, bucket: str, prefix: str, action: str) -> None:
    if isinstance(exc, LoginRefreshRequired):
        raise S3ExperimentError(AWS_LOGIN_MESSAGE) from exc
    if isinstance(exc, ClientError):
        raise S3ExperimentError(_format_client_error(exc, bucket=bucket, prefix=prefix)) from exc
    if isinstance(exc, BotoCoreError):
        raise S3ExperimentError(f"Failed to {action} S3: {exc}") from exc
    raise exc


def list_folder_prefixes(bucket: str, prefix: str = "") -> list[str]:
    """List immediate child folder prefixes under an S3 prefix."""
    normalized_prefix = _normalize_prefix(prefix)
    client = _create_s3_client()
    paginator = client.get_paginator("list_objects_v2")

    folders: list[str] = []
    try:
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=normalized_prefix,
            Delimiter="/",
        ):
            for common_prefix in page.get("CommonPrefixes", []):
                folder_prefix = common_prefix["Prefix"]
                folder_name = folder_prefix[len(normalized_prefix) :].rstrip("/")
                if folder_name:
                    folders.append(folder_name)
    except Exception as exc:
        _handle_s3_exception(exc, bucket=bucket, prefix=prefix, action="list")

    return sorted(folders)


def _iter_objects(client: Any, bucket: str, normalized_prefix: str) -> Iterator[S3ObjectInfo]:
    """Yield every file (not folder marker) under a normalized prefix."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for item in page.get("Contents", []):
            if item["Key"].endswith("/"):
                continue
            yield S3ObjectInfo(
                key=item["Key"],
                size_bytes=int(item.get("Size", 0)),
                last_modified=item["LastModified"].isoformat(),
                etag=item.get("ETag", ""),
            )


def list_objects(
    bucket: str,
    prefix: str,
    *,
    max_keys: int = 500,
) -> list[S3ObjectInfo]:
    """List up to max_keys files under an S3 prefix."""
    normalized_prefix = _normalize_prefix(prefix)
    client = _create_s3_client()

    objects: list[S3ObjectInfo] = []
    try:
        for obj in _iter_objects(client, bucket, normalized_prefix):
            if len(objects) >= max_keys:
                break
            objects.append(obj)
    except Exception as exc:
        _handle_s3_exception(exc, bucket=bucket, prefix=prefix, action="list")

    return objects


def _load_manifest(destination: Path) -> dict[str, dict[str, Any]]:
    path = destination / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.warning("Ignoring unreadable cache manifest at %s", path)
        return {}
    if not isinstance(raw, dict):
        LOGGER.warning("Ignoring malformed cache manifest at %s", path)
        return {}
    return {
        key: entry
        for key, entry in raw.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("etag"), str)
        and isinstance(entry.get("size"), int)
    }


def _save_manifest(destination: Path, manifest: dict[str, dict[str, Any]]) -> None:
    path = destination / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _prune_empty_dirs(directory: Path, stop: Path) -> None:
    """Remove empty folders from directory upwards, never removing stop itself."""
    stop = stop.resolve()
    directory = directory.resolve()
    while directory != stop and directory.is_relative_to(stop) and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def _remove_deleted_files(
    destination: Path, manifest: dict[str, dict[str, Any]], in_s3: set[str]
) -> list[str]:
    """Delete cached files that left S3, but only if they are still what we downloaded."""
    removed: list[str] = []
    for relative_key in sorted(set(manifest) - in_s3):
        entry = manifest.pop(relative_key)
        local_path = _download_path(destination, relative_key)
        if (
            local_path is None
            or not local_path.is_file()
            or local_path.stat().st_size != entry["size"]
        ):
            LOGGER.warning("Not removing %s: it no longer matches the S3 copy", relative_key)
            continue
        local_path.unlink()
        _prune_empty_dirs(local_path.parent, destination)
        removed.append(relative_key)
    return removed


def _download_path(destination: Path, relative_key: str) -> Path | None:
    """Return the local path for a key, or None if it escapes the destination."""
    if relative_key == MANIFEST_NAME:
        return None
    try:
        local_path = resolve_within(destination, relative_key)
    except ValueError:
        return None
    return None if local_path == destination.resolve() else local_path


def download_prefix(
    bucket: str,
    prefix: str,
    destination: Path,
    *,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download all objects under an S3 prefix into a local directory."""
    normalized_prefix = _normalize_prefix(prefix)
    client = _create_s3_client()

    downloaded: list[str] = []
    skipped: list[str] = []
    blocked: list[str] = []
    removed: list[str] = []
    missing_in_s3: list[str] = []
    total_bytes = 0

    try:
        objects = {
            obj.key[len(normalized_prefix) :]: obj
            for obj in _iter_objects(client, bucket, normalized_prefix)
        }
        if not objects:
            removed_count = 0
            if not skip_existing and destination.is_dir():
                manifest = _load_manifest(destination)
                removed_count = len(_remove_deleted_files(destination, manifest, set()))
                _save_manifest(destination, manifest)
            raise ExperimentNotFoundError(bucket, prefix, removed_count)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = _load_manifest(destination)

        for relative_key, obj in objects.items():
            local_path = _download_path(destination, relative_key)
            if local_path is None:
                LOGGER.warning("Blocked unsafe S3 key: s3://%s/%s", bucket, obj.key)
                blocked.append(obj.key)
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)

            entry = manifest.get(relative_key, {})
            unchanged = (
                entry.get("etag") == obj.etag
                and local_path.exists()
                and local_path.stat().st_size == obj.size_bytes
            )
            if skip_existing and unchanged:
                skipped.append(relative_key)
                continue

            LOGGER.info("Downloading s3://%s/%s -> %s", bucket, obj.key, local_path)
            client.download_file(bucket, obj.key, str(local_path))
            manifest[relative_key] = {"etag": obj.etag, "size": obj.size_bytes}
            _save_manifest(destination, manifest)
            downloaded.append(relative_key)
            total_bytes += obj.size_bytes

        if skip_existing:
            missing_in_s3 = sorted(set(manifest) - set(objects))
        else:
            removed = _remove_deleted_files(destination, manifest, set(objects))
        _save_manifest(destination, manifest)
    except Exception as exc:
        _handle_s3_exception(exc, bucket=bucket, prefix=prefix, action="download from")

    return {
        "bucket": bucket,
        "prefix": normalized_prefix,
        "local_path": str(destination),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "blocked_keys": blocked,
        "removed_files": removed,
        "missing_in_s3": missing_in_s3,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "blocked_count": len(blocked),
        "removed_count": len(removed),
        "missing_in_s3_count": len(missing_in_s3),
        "total_bytes": total_bytes,
    }


def iter_relative_keys(keys: Iterable[str], root_prefix: str) -> list[str]:
    """Convert absolute S3 keys to paths relative to an experiment prefix."""
    normalized_prefix = _normalize_prefix(root_prefix)
    relative_paths: list[str] = []
    for key in keys:
        if key.startswith(normalized_prefix):
            relative_paths.append(key[len(normalized_prefix) :])
        else:
            relative_paths.append(key)
    return relative_paths
