"""What each file in an experiment is for, and which folders are runs (docs/adr/0003).

The rules live in ``layout_rules.yaml`` next to this module. A project with other
layouts points ``EXPERIMENT_EXPLORER_LAYOUT_RULES`` at its own rules file, whose roles
are checked first.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiment_explorer.config import get_layout_rules_path
from experiment_explorer.paths import MANIFEST_NAME

ROLES = (
    "report",
    "metrics",
    "predictions",
    "config",
    "figure",
    "code",
    "model",
    "log",
    "data",
    "other",
)
OTHER = "other"
BUILTIN_RULES_PATH = Path(__file__).with_name("layout_rules.yaml")


@dataclass(frozen=True)
class RunFolder:
    """A run folder: its path inside the experiment, its kind and whether it is starred."""

    path: str
    kind: str
    starred: bool


@dataclass(frozen=True)
class LayoutRules:
    roles: tuple[tuple[str, tuple[str, ...]], ...]
    run_name: re.Pattern[str]
    run_kinds: frozenset[str]
    sources: tuple[str, ...]

    def role_of(self, relative_path: str) -> str:
        """The role of a file, from its path inside the experiment."""
        candidate = "/" + relative_path.lower()
        for role, patterns in self.roles:
            if any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns):
                return role
        return OTHER

    def run_folder(self, relative_path: str) -> RunFolder | None:
        """The run a folder is, or None if its name does not make it a run."""
        match = self.run_name.match(Path(relative_path).name)
        if match is None:
            return None
        kind = match.group("kind").lower()
        if kind not in self.run_kinds:
            return None
        return RunFolder(relative_path, kind, bool(match.groupdict().get("star")))


def load_rules() -> LayoutRules:
    """The built-in rules, preceded by the project's rules file if one is configured."""
    sources = [BUILTIN_RULES_PATH]
    extra = get_layout_rules_path()
    if extra is not None:
        sources.insert(0, extra)

    roles: list[tuple[str, tuple[str, ...]]] = []
    run_folder: dict[str, Any] | None = None
    for path in sources:
        content = _read_rules_file(path)
        roles.extend(content["roles"])
        if run_folder is None and content["run_folder"] is not None:
            run_folder = content["run_folder"]
    if run_folder is None:
        raise ValueError("No layout rules file defines run_folder.")
    return LayoutRules(
        roles=tuple(roles),
        run_name=run_folder["name_pattern"],
        run_kinds=run_folder["kinds"],
        sources=tuple(str(path) for path in sources),
    )


def _read_rules_file(path: Path) -> dict[str, Any]:
    """Read and check one rules file; raise ValueError naming the file if it is wrong."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ValueError(f"Cannot read layout rules file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Layout rules file {path} is not valid YAML: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"Layout rules file {path} must be a mapping.")

    roles = []
    for index, rule in enumerate(content.get("roles") or []):
        if not isinstance(rule, dict) or rule.get("role") not in ROLES:
            raise ValueError(
                f"Rule {index} in {path} needs a role, one of {', '.join(ROLES)}."
            )
        patterns = rule.get("match")
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            raise ValueError(f"Rule {index} in {path} needs match: a list of glob patterns.")
        roles.append((rule["role"], tuple(pattern.lower() for pattern in patterns)))

    run_folder = content.get("run_folder")
    if run_folder is not None:
        try:
            name_pattern = re.compile(run_folder["name_pattern"])
            kinds = frozenset(str(kind).lower() for kind in run_folder["kinds"])
        except (KeyError, TypeError, re.error) as exc:
            raise ValueError(
                f"run_folder in {path} needs name_pattern (a regex with a 'kind' group) "
                f"and kinds (a list): {exc}"
            ) from exc
        if "kind" not in name_pattern.groupindex:
            raise ValueError(f"run_folder name_pattern in {path} needs a (?P<kind>...) group.")
        run_folder = {"name_pattern": name_pattern, "kinds": kinds}
    return {"roles": roles, "run_folder": run_folder}


def walk_files(experiment_dir: Path) -> list[tuple[str, Path]]:
    """Return (relative path, path) for every file inside the experiment, sorted.

    Symlinked folders are not followed, and symlinked files that point outside the
    experiment are left out. The cache manifest is bookkeeping, not experiment data.
    """
    root = experiment_dir.resolve()
    found: list[tuple[str, Path]] = []
    for folder, _, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            relative_path = path.relative_to(root).as_posix()
            if relative_path == MANIFEST_NAME:
                continue
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            found.append((relative_path, path))
    return sorted(found)


def find_run_folders(
    experiment_dir: Path, start: Path | None = None, rules: LayoutRules | None = None
) -> list[RunFolder]:
    """Every run folder under ``start`` (default: the whole experiment), sorted by path.

    Symlinked folders are not followed. Runs nested inside other runs are included.
    """
    rules = rules or load_rules()
    base = experiment_dir.resolve()
    runs: list[RunFolder] = []
    for folder, subfolders, _ in os.walk(start or base):
        for name in subfolders:
            path = Path(folder) / name
            if path.is_symlink():
                continue
            run = rules.run_folder(path.relative_to(base).as_posix())
            if run is not None:
                runs.append(run)
    return sorted(runs, key=lambda run: run.path)
