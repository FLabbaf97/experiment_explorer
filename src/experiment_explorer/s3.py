"""S3 helpers for listing and downloading experiment folders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.exceptions import BotoCoreError, ClientError, LoginRefreshRequired

from experiment_explorer.config import get_aws_region

LOGGER = logging.getLogger(__name__)
AWS_LOGIN_MESSAGE = "AWS session expired. Reauthenticate with `aws login` and retry."


class S3ExperimentError(RuntimeError):
    """Raised when S3 operations for experiments fail."""


@dataclass(frozen=True)
class S3ObjectInfo:
    """Metadata for a single S3 object."""

    key: str
    size_bytes: int
    last_modified: str


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


def list_objects(
    bucket: str,
    prefix: str,
    *,
    max_keys: int = 500,
) -> list[S3ObjectInfo]:
    """List files under an S3 prefix."""
    normalized_prefix = _normalize_prefix(prefix)
    client = _create_s3_client()
    paginator = client.get_paginator("list_objects_v2")

    objects: list[S3ObjectInfo] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue
                objects.append(
                    S3ObjectInfo(
                        key=key,
                        size_bytes=int(item.get("Size", 0)),
                        last_modified=item["LastModified"].isoformat(),
                    )
                )
                if len(objects) >= max_keys:
                    return objects
    except Exception as exc:
        _handle_s3_exception(exc, bucket=bucket, prefix=prefix, action="list")

    return objects


def download_prefix(
    bucket: str,
    prefix: str,
    destination: Path,
    *,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download all objects under an S3 prefix into a local directory."""
    normalized_prefix = _normalize_prefix(prefix)
    destination.mkdir(parents=True, exist_ok=True)
    client = _create_s3_client()

    downloaded: list[str] = []
    skipped: list[str] = []
    total_bytes = 0

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue

                relative_key = key[len(normalized_prefix) :]
                local_path = destination / relative_key
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if (
                    skip_existing
                    and local_path.exists()
                    and local_path.stat().st_size == item.get("Size", 0)
                ):
                    skipped.append(relative_key)
                    continue

                LOGGER.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
                client.download_file(bucket, key, str(local_path))
                downloaded.append(relative_key)
                total_bytes += int(item.get("Size", 0))
    except Exception as exc:
        _handle_s3_exception(exc, bucket=bucket, prefix=prefix, action="download from")

    return {
        "bucket": bucket,
        "prefix": normalized_prefix,
        "local_path": str(destination),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
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
