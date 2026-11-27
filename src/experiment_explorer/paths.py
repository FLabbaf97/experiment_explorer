"""Safe experiment names and paths for local cache access."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

# Records the S3 ETag and size of each downloaded file, so changes are detected.
# It lives in each cached experiment folder but is bookkeeping, not experiment data.
MANIFEST_NAME = ".s3_manifest.json"


def validate_experiment_name(name: str) -> str:
    """Return an experiment name without its trailing slash, or raise ValueError.

    Rejects blank, root, absolute, and dot-segment names. Nested names such as
    ``parent/experiment`` remain valid because S3 prefixes can be nested.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Experiment name must not be blank.")
    if name != name.strip() or "\\" in name or "\0" in name:
        raise ValueError(f"Invalid experiment name: {name!r}")
    if name.startswith("/") or PureWindowsPath(name).drive:
        raise ValueError(f"Experiment name must be relative: {name!r}")

    normalized = name.rstrip("/")
    if any(not part.strip() or part in {".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Invalid experiment name: {name!r}")
    return normalized


def resolve_within(root: Path, relative_path: str | Path) -> Path:
    """Resolve a path and raise ValueError if it falls outside root."""
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Refusing path outside {resolved_root}: {relative_path}")
    return candidate


def experiment_cache_path(cache_dir: Path, experiment_name: str) -> Path:
    """Return the validated local cache path for an experiment."""
    return resolve_within(cache_dir, validate_experiment_name(experiment_name))
